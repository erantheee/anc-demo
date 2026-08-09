"""实时 ANC 引擎：block 级自适应谐波消除（自参考，无需参考麦克风）。

M2 现场 demo 的核心：用一支误差麦克风（USB 麦）采噪声，在树莓派上实时
估计基频并对各谐波做 NLMS 幅度/相位跟踪，输出反相波形到扬声器，在误差麦处
形成安静区。对步进电机 / 风扇叶片等周期噪声有效。

- `BlockHarmonicCanceller`：向量化 block 处理（替代逐样本 step，48 kHz 实时可跑）。
- `LiveANCEngine`：sounddevice 双工流线程，分两段采集（ANC off 基线 / ANC on 残差），
  线程安全 `status()` 供仪表盘轮询；`synthetic=True` 时无硬件自测。
- 防啸叫：输出增益有界（tanh + 硬限幅 + 输出/输入 RMS 比例门控 + 权重范数上限）
  + 规则式 watchdog 实时降/升增益，xrun 时丢弃 block 仅推进相位游标
  （避免把错乱时序喂给 NLMS）。
- 反馈中和：用写指针环形缓冲精确读取"delay 前"的输出，从误差麦信号中减掉
  扬声器回声，让 NLMS 只跟踪环境噪声（早期 roll+[len-delay] 读错延迟，
  是降噪变增噪的根因）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.analyze import rms_db


class BlockHarmonicCanceller:
    """向量化自适应谐波消除。每个谐波一组 cos/sin 权重，block NLMS 更新。"""

    def __init__(self, fs: float, f0: float, max_harmonics: int = 10,
                 mu: float = 1e-3, block: int = 512, output_gain: float = 0.5,
                 max_weight_norm: float = 2.0,
                 speaker_mic_delay_s: float = 0.005):
        self.fs = float(fs)
        self.f0 = float(f0)
        self.max_harmonics = int(max_harmonics)
        self.mu = float(mu)
        self.block = int(block)
        self.output_gain = float(output_gain)
        self.max_weight_norm = float(max_weight_norm)
        self.speaker_mic_delay_s = float(speaker_mic_delay_s)
        # 扬声器→误差麦声学延迟（样本数）。输出需"向前预测"这么多样本，
        # 使此刻播出的反相波到达误差麦时恰好对准那时的噪声相位。
        self.predict_ahead_samples = int(round(self.speaker_mic_delay_s * self.fs))
        self.w = np.zeros(2 * self.max_harmonics)
        self._n = 0  # 采样点游标（秒 = n/fs）
        self._basis = self._build_basis()

    def _basis_at(self, start: int) -> np.ndarray:
        """cos/sin 基函数在采样区间 [start, start+block) 的矩阵。"""
        t = (start + np.arange(self.block)) / self.fs
        basis = np.zeros((self.block, 2 * self.max_harmonics))
        for k in range(1, self.max_harmonics + 1):
            ph = 2.0 * np.pi * self.f0 * k * t
            basis[:, 2 * (k - 1)] = np.cos(ph)
            basis[:, 2 * (k - 1) + 1] = np.sin(ph)
        return basis

    def _build_basis(self) -> np.ndarray:
        basis = self._basis_at(self._n)
        self._n += self.block
        return basis

    def _build_output_basis(self) -> np.ndarray:
        """输出重建基函数：相位向前推 predict_ahead_samples 个采样点。

        延迟符号约定（关键）：扬声器在时刻 t 播放的声音经过 delay 秒才到达
        误差麦（时刻 t + delay）。因此反相波必须对准"到达时刻"的噪声，即此刻
        输出应抵消噪声(t + delay)，也就是把噪声相位"向前预测" delay →
        predict_ahead_samples = +delay·fs，输出基函数取 n + predict_ahead_samples。
        若取反（向后预测）则会去抵消 delay 之前的噪声，到达误差麦时相位错开 2·delay，
        反而更差（离线仿真与单测验证了该方向）。
        NLMS 更新仍用当前相位跟踪噪声：对周期信号权重即傅里叶系数，与观测相位
        无关，因此只需在输出侧平移基函数，更新公式不变。

        注意：_n 在每次 _build_basis() 后已指向"下一 block 起点"，故当前 block
        起点是 _n - block；输出预测基必须从当前 block 起点平移，否则会多移一整 block。
        """
        block_start = self._n - self.block
        return self._basis_at(block_start + self.predict_ahead_samples)

    def skip_block(self) -> None:
        """不更新权重，仅推进相位游标（xrun 丢 block 时保持与音频时钟对齐）。"""
        self._basis = self._build_basis()

    def set_predict_ahead_samples(self, n: int) -> None:
        """动态调整输出超前补偿（延迟自动扫描用），不改动已收敛权重。"""
        self.predict_ahead_samples = int(n)

    def process_block(self, desired: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """desired: [block] 误差麦 block。返回 (y_out, e_block)。

        y_out 用预测基重建（补偿扬声器→麦延迟），供反相输出；
        e = desired - y_now 仍是当前相位下的 NLMS 误差（更新用）。
        """
        X = self._basis
        d = np.asarray(desired, dtype=np.float64)[: self.block]
        if len(d) < self.block:
            d = np.pad(d, (0, self.block - len(d)))
        y = X @ self.w
        e = d - y
        # 每路 NLMS block 更新（向量化）
        power = np.sum(X ** 2, axis=0)
        grad = X.T @ e
        self.w = self.w + self.mu * grad / (power + 1e-10)
        # 权重范数上限：相位失步 / 回声正反馈时权重会发散，超过则按比例缩回
        n = float(np.linalg.norm(self.w))
        if n > self.max_weight_norm:
            self.w *= self.max_weight_norm / n
        if self.predict_ahead_samples:
            y_out = self._build_output_basis() @ self.w
        else:
            y_out = y
        self._basis = self._build_basis()
        return y_out, e


def compute_safe_output(y: np.ndarray, in_rms: float, gain: float,
                        clip_level: float = 0.6, ratio: float = 2.5,
                        quiet_rms: float = 1e-3) -> np.ndarray:
    """反相输出防啸叫整形。

    - 输入能量极低（安静）时输出静音，避免向安静环境喷出自身噪声。
    - tanh 压缩后硬限幅到 ±clip_level：**绝对硬上限，与输入无关**——无论输入
      多大、权重如何发散，输出峰值都不可能超过 ±clip_level。
    - 输出 RMS 不得超过 ratio × 输入 RMS（正反馈自激时按比例缩回），
      让输出"跟随"输入：输入小声→反相输出同步小声。
      注意：真正抵消噪声需要输出 ≈ 噪声 / 声耦合，即输出通常要大于输入，
      因此 ratio 必须 > 1 才能产生真实降噪（早期 0.12/1.0 把反相波掐死，
      是"降噪无效果"的原因之一）。
    """
    if in_rms < quiet_rms:
        return np.zeros_like(y)
    out = np.clip(-gain * np.tanh(y), -clip_level, clip_level)
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    max_allowed = ratio * max(in_rms, 1e-12)
    if out_rms > max_allowed:
        out *= max_allowed / out_rms
    return out


def highpass_filter(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    """输入高通：去掉风噪占主导的低频底（DSP 防风第一道）。"""
    from scipy import signal as _signal

    sos = _signal.butter(order, max(cutoff, 1.0), btype="high", fs=fs, output="sos")
    return _signal.sosfilt(sos, np.asarray(x, dtype=np.float64))


def lf_energy_ratio(x: np.ndarray, fs: float, cutoff: float = 100.0) -> float:
    """低频能量占比：<cutoff 功率 / 总功率。

    风噪检测特征：湍流集中在 <100–200Hz，而打印机/风扇谐波噪声、人声的
    能量集中在更高的频带，所以这个比值能把"纯风噪"与"周期噪声/人声"分开。
    cutoff 必须取 ~100Hz：取 200Hz 时打印机 120Hz 基频和人声浊音会被误判为风。
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 64:
        return 0.0
    X = np.abs(np.fft.rfft(x - np.mean(x))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    total = float(np.sum(X))
    if total < 1e-12:
        return 0.0
    return float(np.sum(X[f < cutoff]) / total)


def differential_wind_ratio(l: np.ndarray, r: np.ndarray, fs: float,
                            lo_hz: float = 20.0, hi_hz: float = 4000.0) -> float:
    """双麦差分风噪指标：频带 [lo,hi] 内差分能量 / 和能量。

    原理（已用 Pi USB 双 capsule 实测验证）：
    - 远场相干声（电机/风扇/打印机/人声）：到达紧邻两 capsule 几乎同相 →
      L≈R → 差分≈0，比值接近 0；
    - 风噪是近场湍流：两 capsule 听到的是不同涡 → L≠R → 差分≈单路能量，
      比值接近 1 甚至更高。

    因此用「差分 RMS / 和 RMS」做风噪指示：接近 0 = 相干周期声，接近 1+ =
    风噪主导。比单麦 lf_energy_ratio 更鲁棒：50Hz 哼声/低频音调在两路相干，
    不会误判为风。

    频带选择（真机标定结论）：
    - lo_hz 必须取 ~20Hz：风的湍流能量集中在 20–60Hz（卡门涡街/阵风），
      用 60Hz 会把风噪特征整个切掉，差分指标退化为噪声本底；
    - hi_hz 取 4kHz 足够：>4kHz 两路对远场声也因 capsule 间距开始失配，
      会引入假差分。
    """
    l = np.asarray(l, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    n = min(len(l), len(r))
    if n < 64:
        return 0.0
    l, r = l[:n] - l[:n].mean(), r[:n] - r[:n].mean()
    d = l - r
    s = (l + r) / 2.0
    f = np.fft.rfftfreq(n, 1 / fs)
    m = (f >= lo_hz) & (f < hi_hz)
    p_d = float(np.sum(np.abs(np.fft.rfft(d)[m]) ** 2))
    p_s = float(np.sum(np.abs(np.fft.rfft(s)[m]) ** 2))
    if p_s < 1e-12:
        return 0.0
    return float(np.sqrt(p_d / p_s))


@dataclass
class LiveState:
    state: str = "idle"  # idle | running | stopping | stopped | error
    phase: str = "idle"  # idle | baseline | cancelling | done
    f0: float | None = None
    gain: float = 0.5
    mic_delay_ms: float = 5.0  # 扬声器→误差麦延迟补偿（当前生效值）
    delay_scan_ms: float | None = None  # 延迟自动扫描选中的补偿（ms）
    spl_now_db: float | None = None
    baseline_spl_db: float | None = None
    cancelling_spl_db: float | None = None
    reduction_db: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None
    # 规则式 watchdog（实时防啸叫 / 降噪不足自动调增益）
    watchdog_enabled: bool = True
    watchdog_log: list = field(default_factory=list)   # 最近调节日志（最多 20 条）
    watchdog_reduce_count: int = 0   # 自动降增益次数
    watchdog_increase_count: int = 0  # 自动增增益次数
    # 风噪门控（实时）：低频能量占比超阈值时静音反相输出
    wind_gate_enabled: bool = False
    wind_mute: bool = False
    wind_ratio_now: float | None = None
    wind_diff_ratio_now: float | None = None  # 双麦差分风噪指标（dual_mic 时）


def _pick_duplex_default() -> int | None:
    """选择可同时录放的双工设备索引。

    Linux 上常通过 ~/.asoundrc 配置 asym 双工设备（如 `default` / `anc_duplex`），
    让单个 USB 音频设备在 PortAudio 下也能开全双工流。优先按名字匹配，避免
    设备被占用时通道数误报为 0。
    """
    try:
        import sounddevice as sd
    except ImportError:
        return None
    try:
        for i, d in enumerate(sd.query_devices()):
            if d["name"] in ("default", "anc_duplex"):
                return i
    except Exception:
        return None
    return None


class LiveANCEngine:
    """实时 ANC 环路线程。无参考麦克风：从误差信号自参考估计并消除谐波。

    # 实时环路数据流（音频回调线程，每 block 一次）
    #
    #   d(误差麦) ─► 反馈中和 ─► NLMS 消除 ─► y ─► compute_safe_output ─► 扬声器
    #        │              ▲                                        │
    #        └── e_buf ◄────┴──────── 误差麦回采残差 ◄────────────────┘
    #
    # 规则式 watchdog 双规则（每 ~0.5s 评估，不依赖 LLM，作为 Agent 调节的兜底）：
    #
    #   ┌─ SPL 上升斜率 > growth_thresh (3 dB/s) ？──是──► 降增益（疑似啸叫）
    #   │        │否
    #   │        └─ SPL 未低于 baseline - 1.5dB ？──是──► 增增益（上限 1.0）
    #   │                │否
    #   │                └──────────────────────────────► 不动
    #
    # 延迟自动扫描（cancelling 开始）：依次试 20/30/40/50/60ms 补偿，
    # 每档采样 ~0.4s，取误差麦 RMS 最小者；期间输出减半压低试错爆音。
    """

    def __init__(self, fs: int = 48000, in_device=None, out_device=None,
                 block: int = 512, f0: float | None = None, max_harmonics: int = 10,
                 mu: float = 2e-2, output_gain: float = 0.5, baseline_s: float = 5.0,
                 max_duration_s: float = 60.0, synthetic: bool = False,
                 echo_gain: float = 0.0,
                 speaker_mic_delay_s: float = 0.005,
                 feedback_cancel_gain: float = 0.3,
                 auto_scan_delay: bool = True,
                 watchdog_enabled: bool = True,
                 watchdog_growth_thresh_db_s: float = 3.0,
                 watchdog_min_gain: float = 0.02,
                 watchdog_max_gain: float = 1.0,
                 watchdog_increase_step: float = 0.05,
                 watchdog_reduction_target_db: float = 1.5,
                 wind_gate_enabled: bool = False,
                 wind_gate_cutoff_hz: float = 100.0,
                 wind_gate_ratio_thresh: float = 0.8,
                 wind_gate_window_s: float = 0.25,
                 input_highpass_hz: float = 0.0,
                 dual_mic: bool = False,
                 wind_gate_diff_thresh: float = 0.6) -> None:
        self.fs = int(fs)
        self.in_device = in_device
        self.out_device = out_device
        self.block = int(block)
        self.f0_init = f0
        self.max_harmonics = int(max_harmonics)
        self.mu = float(mu)
        self.output_gain = float(output_gain)
        self.baseline_s = float(baseline_s)
        self.max_duration_s = float(max_duration_s)
        self.synthetic = bool(synthetic)
        self.echo_gain = float(echo_gain)
        self.speaker_mic_delay_s = float(speaker_mic_delay_s)
        # 反馈中和：从麦克风信号中减去"扬声器回声估计"，避免 NLMS 追自己的输出
        self.feedback_cancel_gain = float(feedback_cancel_gain)
        # 延迟自动扫描：cancelling 开始时试几个偏移，取误差麦 RMS 最小者
        self.auto_scan_delay = bool(auto_scan_delay)
        self._scan_offsets_ms = [20.0, 30.0, 40.0, 50.0, 60.0]
        # 规则式 watchdog：实时防啸叫 + 降噪不足自动调增益
        self.watchdog_enabled = bool(watchdog_enabled)
        self.watchdog_growth_thresh_db_s = float(watchdog_growth_thresh_db_s)
        self.watchdog_min_gain = float(watchdog_min_gain)
        self.watchdog_max_gain = float(watchdog_max_gain)
        self.watchdog_increase_step = float(watchdog_increase_step)
        self.watchdog_reduction_target_db = float(watchdog_reduction_target_db)
        # 风噪门控：低频能量占比超阈值 → 静音反相输出（阻止对不相关风喷噪声）
        self.wind_gate_enabled = bool(wind_gate_enabled)
        self.wind_gate_cutoff_hz = float(wind_gate_cutoff_hz)
        self.wind_gate_ratio_thresh = float(wind_gate_ratio_thresh)
        self.wind_gate_window_s = float(wind_gate_window_s)
        # 输入高通：去掉风噪主导的低频底，防止 NLMS 把权重耗在追风上
        self.input_highpass_hz = float(input_highpass_hz)
        self._hp_buf: np.ndarray | None = None  # sosfilt 的持续状态（跨 block 保持）
        # 双麦差分风噪检测：第二路 capsule 当参考麦，差分能量/和能量驱动门控
        self.dual_mic = bool(dual_mic)
        self.wind_gate_diff_thresh = float(wind_gate_diff_thresh)
        # watchdog 实时状态（回调线程写，status 读）
        self._wd_spl_history: list[float] = []
        self._wd_last_growth_db_s: float = 0.0
        self._wd_blocks_since_check = 0
        self._wd_check_every_blocks = max(1, int(0.5 * self.fs / max(self.block, 1)))
        self._wd_last_increase_ts = 0.0
        # 规则 2a 因果验证：记录"降增益前"的 SPL 与增益，验证降增益是否真的
        # 让误差麦 SPL 下降。若 SPL 与增益无关（外部噪声源变化 / 声耦合主导），
        # 恢复增益并冻结 2a/2b 一段时间，避免误把增益砍没导致无法降噪。
        self._wd_last_reduce_ts = 0.0
        self._wd_reduce_spl_before: float | None = None
        self._wd_reduce_gain_before: float | None = None
        self._wd_reduce_check_count = 0
        self._wd_rule2ab_freeze_until = 0.0
        self._wd_log_lock = threading.Lock()

        self.state = LiveState(watchdog_enabled=self.watchdog_enabled,
                               wind_gate_enabled=self.wind_gate_enabled)
        self.d_buf: list[np.ndarray] = []   # 基线（ANC off）
        self.e_buf: list[np.ndarray] = []   # 残差（ANC on，误差麦真实信号）
        self._canceller: BlockHarmonicCanceller | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts = 0.0
        # 反馈中和环形缓冲（写指针方案）：读 (写指针 - delay) % len 精确取 delay 前的输出。
        # 长度必须取 block 的整数倍：_out_pos 按 block 步进，若 len 非 block 倍数，
        # 写切片 out_hist[pos:pos+block] 在绕回处会截断 → 真机回调 ~0.2s 即崩溃。
        _buf_len = max(8192, int(0.2 * fs), block)
        _buf_len = (_buf_len // block) * block
        self._out_hist = np.zeros(_buf_len, dtype=np.float64)
        self._out_pos = 0  # 写指针：下一 block 的写入位置
        # 延迟扫描状态
        self._scan_active = False
        self._scan_idx = 0
        self._scan_acc = 0.0
        self._scan_blocks = 0
        self._scan_scores: list[float] = []
        # 人声门控（实时）：检测到说话声时静音反相输出，避免反相波"吃掉"人声。
        # 回调线程只做"收集信号 + 读 mute 标志"，真正的 FFT 语音检测放主循环
        # （_voice_gate_tick），避免把重计算塞进音频回调引起 xrun。
        self._voice_hist: list[np.ndarray] = []
        self._voice_buf_max = max(int(1.2 * self.fs / max(self.block, 1)), 4)
        self._voice_mute = False
        self._voice_clear_s = 0.0
        self._voice_reasons: list[str] = []
        self._voice_check_interval_s = 0.5
        self._voice_last_check = 0.0
        # 风噪门控状态（__init__ 初始化，start 时重置；供门控 tick 与回调读取）
        self._wind_hist: list[np.ndarray] = []
        self._wind_buf_max = max(int(self.wind_gate_window_s * self.fs / max(self.block, 1)) + 2, 4)
        self._wind_mute = False
        self._wind_ratio_now: float | None = None
        self._wind_diff_ratio_now: float | None = None

    # ---- 控制 ----

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"started": False, "message": "已在运行"}
        with self._lock:
            self.state = LiveState(state="running", phase="baseline",
                                   gain=self.output_gain, f0=self.f0_init,
                                   mic_delay_ms=self.speaker_mic_delay_s * 1000.0,
                                   watchdog_enabled=self.watchdog_enabled,
                                   wind_gate_enabled=self.wind_gate_enabled)
            self.state.delay_scan_ms = None
            self.d_buf = []
            self.e_buf = []
            self._canceller = None
            self._start_ts = time.time()
            # 重置 watchdog 状态
            self._wd_spl_history = []
            self._wd_blocks_since_check = 0
            self._wd_last_increase_ts = 0.0
            self._wd_last_reduce_ts = 0.0
            self._wd_reduce_spl_before = None
            self._wd_reduce_gain_before = None
            self._wd_reduce_check_count = 0
            self._wd_rule2ab_freeze_until = 0.0
            # 重置反馈中和环形缓冲（避免上一次运行的残留回声污染新一轮）
            self._out_hist[:] = 0.0
            self._out_pos = 0
            # 重置人声门控状态
            self._voice_hist = []
            self._voice_mute = False
            self._voice_clear_s = 0.0
            self._voice_reasons = []
            self._voice_last_check = 0.0
            # 重置风噪门控与高通滤波状态
            self._wind_hist: list[np.ndarray] = []
            self._wind_buf_max = max(int(self.wind_gate_window_s * self.fs / max(self.block, 1)) + 2, 4)
            self._wind_mute = False
            self._wind_ratio_now: float | None = None
            self._wind_diff_ratio_now: float | None = None
            self._hp_buf = None
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="anc-live", daemon=True)
        self._thread.start()
        return {"started": True}

    def stop(self) -> dict:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=6.0)
        return self.status()

    def status(self) -> dict:
        with self._lock:
            st = self.state
            elapsed = round(float(st.elapsed_s or 0.0), 1)

            def _r(v):
                return round(float(v), 1) if v is not None else None

            return {
                "state": st.state,
                "phase": st.phase,
                "f0": float(st.f0) if st.f0 is not None else None,
                "gain": float(st.gain),
                "mic_delay_ms": _r(st.mic_delay_ms),
                "delay_scan_ms": _r(st.delay_scan_ms),
                "spl_now_db": _r(st.spl_now_db),
                "baseline_spl_db": _r(st.baseline_spl_db),
                "cancelling_spl_db": _r(st.cancelling_spl_db),
                "reduction_db": _r(st.reduction_db),
                "elapsed_s": elapsed,
                "synthetic": self.synthetic,
                "watchdog": {
                    "enabled": bool(st.watchdog_enabled),
                    "log": list(st.watchdog_log),
                    "reduce_count": int(st.watchdog_reduce_count),
                    "increase_count": int(st.watchdog_increase_count),
                },
                "wind_gate": {
                    "enabled": bool(st.wind_gate_enabled),
                    "mute": bool(st.wind_mute),
                    "lf_ratio": round(float(st.wind_ratio_now), 3) if st.wind_ratio_now is not None else None,
                    "diff_ratio": round(float(st.wind_diff_ratio_now), 3) if st.wind_diff_ratio_now is not None else None,
                    "cutoff_hz": float(self.wind_gate_cutoff_hz),
                    "ratio_thresh": float(self.wind_gate_ratio_thresh),
                    "diff_thresh": float(self.wind_gate_diff_thresh),
                    "dual_mic": bool(self.dual_mic),
                },
                "voice_gate": {
                    "mute": bool(self._voice_mute),
                    "reasons": list(self._voice_reasons),
                },
                "error": st.error,
            }

    # ---- 可写控制（Agent / Web 调用） ----

    def control(self, action: str, params: dict | None = None) -> dict:
        """统一控制入口，供 Agent 的 adjust_anc 工具与 Web 手动调节调用。

        action ∈ set_gain | increase_gain | decrease_gain | set_mic_delay_ms | stop。
        返回 {"ok": bool, ...}，成功时附加当前 status()。
        """
        params = params or {}
        try:
            if action == "set_gain":
                value = params.get("value")
                if value is None:
                    return {"ok": False, "error": "缺少 value（目标增益）"}
                return self.set_output_gain(float(value), params.get("reason", ""))
            if action == "increase_gain":
                delta = params.get("delta") if params.get("delta") is not None else 0.02
                return self.step_output_gain(+float(delta), params.get("reason", "Agent 增增益"))
            if action == "decrease_gain":
                delta = params.get("delta") if params.get("delta") is not None else 0.05
                return self.step_output_gain(-float(delta), params.get("reason", "Agent 降增益"))
            if action == "set_mic_delay_ms":
                ms = params.get("mic_delay_ms")
                if ms is None:
                    return {"ok": False, "error": "缺少 mic_delay_ms"}
                return self.set_mic_delay_ms(float(ms), params.get("reason", ""))
            if action == "stop":
                return {"ok": True, **self.stop()}
            return {"ok": False, "error": f"未知控制动作: {action}"}
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def set_output_gain(self, value: float, reason: str = "") -> dict:
        """设置反相输出增益（带安全上限）。实时生效于下一个 block。"""
        with self._lock:
            self._apply_gain(value, reason)
        return {"ok": True, **self.status()}

    def step_output_gain(self, delta: float, reason: str = "") -> dict:
        """按增量调节增益（Agent 啸叫降增益 / 降噪不足增增益用）。

        「读当前 → 计算 → 钳制 → 写回」全部在单次加锁内完成：watchdog（音频
        回调线程）可能与本方法并发改增益，若读与写分两次加锁，会基于过期值
        计算导致本次增量被覆盖丢失。
        """
        with self._lock:
            self._apply_gain(self.state.gain + delta, reason)
        return {"ok": True, **self.status()}

    def _apply_gain(self, value: float, reason: str = "") -> None:
        """锁内应用增益：钳制安全边界 + 写入引擎/状态 + 记日志。调用方须持有 _lock。"""
        value = float(np.clip(value, self.watchdog_min_gain, self.watchdog_max_gain))
        self.output_gain = value
        self.state.gain = value
        if self._canceller is not None:
            self._canceller.output_gain = value
        self._append_wd_log("set_gain", f"增益设为 {value:.2f}" + (f"（{reason}）" if reason else ""))

    def set_mic_delay_ms(self, ms: float, reason: str = "") -> dict:
        """动态调整扬声器→误差麦延迟补偿（毫秒）。实时生效于下一个 block。"""
        ms = float(ms)
        with self._lock:
            self.speaker_mic_delay_s = ms / 1000.0
            self.state.mic_delay_ms = ms
            if self._canceller is not None:
                self._canceller.set_predict_ahead_samples(int(round(ms * self.fs / 1000.0)))
            self._append_wd_log("set_delay", f"延迟补偿设为 {ms:.1f} ms" + (f"（{reason}）" if reason else ""))
        return {"ok": True, **self.status()}

    def _append_wd_log(self, action: str, msg: str) -> None:
        entry = {"t": round(time.time(), 1), "action": action, "msg": msg}
        with self._wd_log_lock:
            log = self.state.watchdog_log
            log.append(entry)
            if len(log) > 20:
                del log[:-20]

    def get_signals(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """返回拼接后的 (baseline d, 残差 e)，供 A/B 评估。"""
        with self._lock:
            d = np.concatenate(self.d_buf) if self.d_buf else None
            e = np.concatenate(self.e_buf) if self.e_buf else None
            return d, e

    def residual_tail(self, n: int) -> np.ndarray | None:
        """返回最近 n 个采样点的残差（ANC 运行中误差麦真实信号）。

        供噪声检测 Agent 在 ANC 运行时直接分析残差里是否藏着啸叫（而不是
        重新采集麦克风，因为运行中麦克风采到的主要是扬声器反相波 + 残余噪声）。
        未运行或尚无残差数据时返回 None。返回副本，不含引用。
        """
        n = int(n)
        if n <= 0:
            return None
        with self._lock:
            if not self.e_buf:
                return None
            chunks: list[np.ndarray] = []
            remaining = n
            for blk in reversed(self.e_buf):
                if remaining <= 0:
                    break
                take = min(len(blk), remaining)
                chunks.append(blk[-take:])
                remaining -= take
        if not chunks:
            return None
        return np.concatenate(chunks[::-1]).copy()

    def _read_echo(self, delay_n: int) -> np.ndarray:
        """读 delay_n 个采样点前写入的输出块（反馈中和用）。

        写指针环形缓冲：缓冲内容按时间顺序排布（写指针前进方向即时间前进
        方向），(写指针 - delay_n) % len 就是 delay_n 前的样本。旧实现用
        np.roll + 头写 + [len-delay_n] 读取，实测把延迟读成了 len-delay_n
        （40ms 目标被读成 ~130ms），中和信号错位 → NLMS 追错残差 → 输出与
        噪声同相叠加，降噪变增噪（Pi 现场 +1.8 dB 的根因）。预热期（写入
        历史不足 delay_n）返回静音，等价于还没有回声。
        """
        n = len(self._out_hist)
        if delay_n <= 0 or delay_n >= n or delay_n + self.block > n:
            return np.zeros(self.block)
        start = (self._out_pos - delay_n) % n
        end = start + self.block
        if end <= n:
            return self._out_hist[start:end]
        return np.concatenate([self._out_hist[start:], self._out_hist[:end - n]])

    # ---- 内部 ----

    def _run(self) -> None:
        try:
            if self.synthetic:
                self._run_synthetic()
            else:
                self._run_audio()
        except Exception as exc:
            with self._lock:
                self.state.state = "error"
                self.state.error = str(exc)
        finally:
            with self._lock:
                if self.state.state != "error":
                    self.state.state = "stopped"

    def _begin_cancel(self) -> None:
        """从已采基线估计 f0，创建消除器并进入 cancelling 相位。"""
        with self._lock:
            if self.state.state == "error":
                return
            buf = self.d_buf
        if not buf:
            return
        d = np.concatenate(buf)
        # 人声闸门：基线像人声时不进入降噪（仅在自动估计基频时启用）。
        # 谐波消除会把说话声当成周期源反向消除，听感上像"吃掉"说话声，
        # 因此人声不得作为 ANC 目标。
        if self.f0_init is None:
            from app.voice import detect_voice
            vd = detect_voice(d, self.fs)
            if vd["is_voice"]:
                with self._lock:
                    self.state.state = "error"
                    self.state.error = (
                        "基线信号疑似人声/语音（%s）。ANC 只对稳态周期噪声"
                        "（电机/风扇/打印机）有效，对说话声会反向消除、像'吃掉'人声。"
                        "请等人声停止，或只对设备噪声启动 ANC。" % "；".join(vd["reasons"])
                    )
                    return
        with self._lock:
            self.state.baseline_spl_db = rms_db(d)
        # 基线信号有效性：麦克风悬空/未接时 RMS 极低，禁止继续（避免"凭空降噪"）
        if self.state.baseline_spl_db is not None and self.state.baseline_spl_db < -60.0:
            raise RuntimeError(
                "基线信号无效（RMS %.0f dBFS，低于可用阈值）。麦克风可能未连接或线缆悬空，"
                "请检查误差麦克风后再试" % self.state.baseline_spl_db
            )
        # 只用尾部 ~1s 估基频：自相关是 O(n²)，整段 3s/48k 会阻塞主循环数秒
        tail = d[-int(self.fs):] if len(d) >= self.fs else d
        f0 = self.f0_init or estimate_fundamental(tail, self.fs)
        # 风噪门控（开启时）：基线低频能量占比过高 → 纯风噪，拒绝伪 f0。
        # 强风噪会把自相关骗出一个"伪基频"（如卡门涡街 40-60Hz 或噪声的偶然
        # 周期），谐波消除器按它输出一整串不相关反相波 → 误差麦处能量平方叠加
        # （"更尖锐且不降噪"）。风噪不是可消除的周期源，直接拒绝进入 cancelling。
        # 仅在 wind_gate_enabled 时生效：不开启的用户（低频周期源如 50Hz 哼声）
        # 行为不变，不会被误拒。
        if (self.wind_gate_enabled and f0 is not None
                and lf_energy_ratio(tail, self.fs,
                                    cutoff=self.wind_gate_cutoff_hz) > self.wind_gate_ratio_thresh):
            with self._lock:
                self.state.state = "error"
                self.state.error = (
                    "基线低频能量占比 %.2f > %.2f（疑似风噪/气流声）。ANC 谐波消除只对"
                    "稳态周期噪声（电机/风扇/打印机）有效，风噪是低频宽带非平稳信号，"
                    "无法反相消除且会输出不相关噪声。请遮挡麦克风进风口或更换消声环境。"
                    % (lf_energy_ratio(tail, self.fs, cutoff=self.wind_gate_cutoff_hz),
                       self.wind_gate_ratio_thresh)
                )
                return
        with self._lock:
            if f0 is None:
                self.state.state = "error"
                self.state.error = "未检测到周期噪声基频，无法谐波消除。可用 --f0 手动指定，或确认噪声源为稳态周期噪声。"
                return
            self.state.f0 = float(f0)
            self._canceller = BlockHarmonicCanceller(
                fs=self.fs, f0=float(f0), max_harmonics=self.max_harmonics,
                mu=self.mu, block=self.block, output_gain=self.output_gain,
                speaker_mic_delay_s=self.speaker_mic_delay_s)
            self.state.phase = "cancelling"
            # 延迟自动扫描：真机模式默认开启，扫描期间用首个偏移试起
            if self.auto_scan_delay and not self.synthetic:
                self._scan_active = True
                self._scan_idx = 0
                self._scan_acc = 0.0
                self._scan_blocks = 0
                self._scan_scores = []
                self._canceller.set_predict_ahead_samples(
                    int(self._scan_offsets_ms[0] * self.fs / 1000.0))

    # ---- 规则式 watchdog ----

    def _watchdog_feed(self, spl_db: float) -> None:
        """每个 block 喂一次误差麦 SPL，周期性评估并自动调节增益。

        两条规则（不依赖 LLM，亚秒级响应，作为 Agent 上层调节的兜底）：
        1. **啸叫 / 正反馈**：误差麦 SPL 持续快速上升（近期窗口线性斜率
           > watchdog_growth_thresh_db_s dB/s）→ 立即降增益。
        2. **降噪不足**：进入 cancelling 一段时间后，当前 SPL 仍明显高于
           baseline - watchdog_reduction_target_db → 适当增增益（有上限）。
        """
        self._wd_spl_history.append(float(spl_db))
        if len(self._wd_spl_history) > 120:
            del self._wd_spl_history[:-120]
        self._wd_blocks_since_check += 1
        if self._wd_blocks_since_check < self._wd_check_every_blocks:
            return
        self._wd_blocks_since_check = 0
        if len(self._wd_spl_history) < 8:
            return

        with self._lock:
            st = self.state
            if st.phase != "cancelling":
                return
            canceller = self._canceller
            if canceller is None:
                return

            # 近期窗口线性斜率 → 每秒 dB 增长率
            y = np.asarray(self._wd_spl_history, dtype=np.float64)
            x = np.arange(len(y), dtype=np.float64)
            slope = np.polyfit(x, y, 1)[0] if len(y) >= 2 else 0.0
            growth_db_s = float(slope / (self.block / self.fs))  # per-second slope
            self._wd_last_growth_db_s = growth_db_s
            cur_spl = float(y[-1])

            # 规则 1：快速上升 = 啸叫 / 正反馈
            if growth_db_s > self.watchdog_growth_thresh_db_s:
                new_gain = max(self.watchdog_min_gain, canceller.output_gain - 0.02)
                if new_gain < canceller.output_gain:
                    canceller.output_gain = new_gain
                    st.gain = new_gain
                    st.watchdog_reduce_count += 1
                    self._append_wd_log(
                        "reduce_gain",
                        f"SPL 上升 {growth_db_s:.1f} dB/s（疑似啸叫），增益 {new_gain:.2f}")
                return

            # 规则 2：降噪不足 → 增增益（进入 cancelling 至少 3s 后）
            baseline = st.baseline_spl_db
            now = time.time()
            if (baseline is not None and st.elapsed_s > self.baseline_s + 3.0):
                # 因果验证：上次因规则 2a 降了增益，隔 3 次检查（≈1.5s）后看
                # SPL 是否真的降下来了。验证完成前不再次触发 2a 降增益。
                if self._wd_reduce_spl_before is not None:
                    self._wd_reduce_check_count += 1
                    if self._wd_reduce_check_count >= 3:
                        spl_before = self._wd_reduce_spl_before
                        gain_before = self._wd_reduce_gain_before or canceller.output_gain
                        self._wd_reduce_spl_before = None
                        self._wd_reduce_gain_before = None
                        self._wd_reduce_check_count = 0
                        if cur_spl >= spl_before - 1.0:
                            if gain_before > canceller.output_gain:
                                canceller.output_gain = gain_before
                                st.gain = gain_before
                                self._append_wd_log(
                                    "restore_gain",
                                    f"降增益无效（SPL {spl_before:.1f}→{cur_spl:.1f} 未降，"
                                    f"疑为外部噪声源/声耦合），恢复增益 {gain_before:.2f}")
                            self._wd_rule2ab_freeze_until = now + 15.0
                            return
                        self._append_wd_log(
                            "reduce_effective",
                            f"降增益有效（SPL {spl_before:.1f}→{cur_spl:.1f}），保持")
                    else:
                        # 验证窗口内：不再次触发 2a/2b，等因果结论
                        return
                else:
                    self._wd_reduce_check_count = 0

                # 冻结期：跳过 2a/2b 自动调节（规则 1 啸叫保护仍生效）
                if now < self._wd_rule2ab_freeze_until:
                    return

                # 规则 2a：当前 SPL 明显高于基线 → ANC 在放大噪声（反相波与噪声
                # 不匹配，如 f0 估计错误 / 相位失准），继续加增益只会更糟。
                # 此时应降增益，把"尖锐放大"压回去。
                if cur_spl > baseline + 6.0:
                    new_gain = max(self.watchdog_min_gain, canceller.output_gain - 0.05)
                    if new_gain < canceller.output_gain:
                        canceller.output_gain = new_gain
                        st.gain = new_gain
                        st.watchdog_reduce_count += 1
                        self._wd_last_reduce_ts = now
                        self._wd_reduce_spl_before = float(cur_spl)
                        self._wd_reduce_gain_before = float(canceller.output_gain + 0.05)
                        self._append_wd_log(
                            "reduce_gain",
                            f"ANC 放大噪声（当前 {cur_spl:.1f} > 基线 {baseline:.1f} dB），"
                            f"降增益 {new_gain:.2f}")
                    return
                # 规则 2b：降噪不足（还没降到目标）→ 增增益（有上限）。
                # 仅在 SPL 处于"接近/略高于基线"的合理区间时增（若还远高于基线
                # 说明要么在放大、要么外部声源主导，增增益无意义，避免与 2a 打架）
                if (baseline - 12.0 <= cur_spl <= baseline + 6.0
                        and cur_spl > baseline - self.watchdog_reduction_target_db
                        and canceller.output_gain < self.watchdog_max_gain
                        and now - self._wd_last_increase_ts > 2.0):
                    new_gain = min(self.watchdog_max_gain,
                                   canceller.output_gain + self.watchdog_increase_step)
                    canceller.output_gain = new_gain
                    st.gain = new_gain
                    st.watchdog_increase_count += 1
                    self._wd_last_increase_ts = now
                    self._append_wd_log(
                        "increase_gain",
                        f"降噪不足（当前 {cur_spl:.1f} vs 基线 {baseline:.1f} dB），增益 {new_gain:.2f}")

    # ---- 内部 ----

    def _voice_gate_tick(self) -> None:
        """周期（~0.5s）人声检测并更新门控：检测到人声则静音反相输出。

        由主循环在回调线程外调用，避免把 FFT 语音检测塞进音频回调引起 xrun。
        需要连续 ~1.2s 误差麦信号，且语音检测得分达标才判定为人声（需要
        2 个及以上指标：f0 波动 / RMS 调制 / 浊音占比，见 app/voice.py）。
        人声持续出现 → 静音；人声停止 ~1.5s 后自动恢复降噪输出。
        """
        now = time.time()
        if now - self._voice_last_check < self._voice_check_interval_s:
            return
        self._voice_last_check = now
        with self._lock:
            buf = list(self._voice_hist)
        is_voice, reasons = False, []
        if len(buf) >= 3:
            from app.voice import detect_voice
            vd = detect_voice(np.concatenate(buf), self.fs)
            is_voice, reasons = vd["is_voice"], vd["reasons"]
        if is_voice:
            if not self._voice_mute:
                self._append_wd_log(
                    "voice_mute",
                    "检测到人声（%s），静音反相输出，避免'吃掉'说话声"
                    % "；".join(reasons))
            self._voice_mute = True
            self._voice_clear_s = 0.0
            self._voice_reasons = list(reasons)
        elif self._voice_mute:
            self._voice_clear_s += self._voice_check_interval_s
            if self._voice_clear_s >= 1.5:
                self._voice_mute = False
                self._voice_reasons = []
                self._append_wd_log("voice_unmute", "人声停止，恢复 ANC 输出")

    def _wind_gate_tick(self) -> None:
        """周期（~0.25s）风噪检测并更新门控：低频能量占比超阈值 → 静音反相输出。

        由主循环在回调线程外调用（与 _voice_gate_tick 同帧），FFT 不进音频回调。
        风噪（湍流低频堆积）的低频占比接近 1，周期噪声/人声远低于阈值。
        cutoff 默认 100Hz：取 200Hz 会被打印机 120Hz 基频和人声浊音误触发。
        """
        if not self.wind_gate_enabled:
            return
        with self._lock:
            buf = list(self._wind_hist)
        if not buf:
            return
        # 双麦差分风噪：两路分别取平均后算差分能量/和能量比（对周期声相干、
        # 对风噪不相关）；否则退回单麦低频能量占比。
        if isinstance(buf[0], tuple):
            l = np.concatenate([b[0] for b in buf])
            r = np.concatenate([b[1] for b in buf])
            ratio = differential_wind_ratio(l, r, self.fs)
            self.state.wind_diff_ratio_now = ratio
            self._wind_diff_ratio_now = ratio
            # 能量门槛：弱信号（两路各自本底噪声）差分天然不相关，会把安静环境
            # 误判为风。只有信号强度足够时差分指标才可信。
            rms = float(np.sqrt(np.mean(l ** 2)))
            if 20.0 * np.log10(max(rms, 1e-12)) < -50.0:
                ratio = 0.0
                self._wind_ratio_now = 0.0
                self.state.wind_ratio_now = 0.0
                self._wind_diff_ratio_now = 0.0
                self.state.wind_diff_ratio_now = 0.0
                with self._lock:
                    self._wind_mute = False
                    self.state.wind_mute = False
                return
            gate = ratio > self.wind_gate_diff_thresh
            thresh = self.wind_gate_diff_thresh
            tag = "差分风噪指标"
        else:
            win = np.concatenate(buf)
            ratio = lf_energy_ratio(win, self.fs, cutoff=self.wind_gate_cutoff_hz)
            self.state.wind_diff_ratio_now = None
            self._wind_diff_ratio_now = None
            gate = ratio > self.wind_gate_ratio_thresh
            thresh = self.wind_gate_ratio_thresh
            tag = "低频占比"
        with self._lock:
            self._wind_ratio_now = ratio
            self.state.wind_ratio_now = ratio
            if gate:
                if not self._wind_mute:
                    self._append_wd_log(
                        "wind_mute",
                        f"风噪检测：{tag} {ratio:.2f} > {thresh:.2f}，"
                        f"静音反相输出（避免对不相关风喷噪声）")
                self._wind_mute = True
                self.state.wind_mute = True
            else:
                if self._wind_mute:
                    self._append_wd_log(
                        "wind_unmute",
                        f"风噪消退：{tag} {ratio:.2f}，恢复 ANC 输出")
                self._wind_mute = False
                self.state.wind_mute = False

    def _hp_filter(self, d: np.ndarray) -> np.ndarray:
        """输入高通（带持续状态）：去掉风噪主导的低频底。

        用 sosfilt 的 zf 状态跨 block 保持滤波连续性（一次性整段滤波在
        block 边界会有瞬态）。返回与输入等长的滤波后数组。
        """
        if self.input_highpass_hz <= 0:
            return d
        from scipy import signal as _signal

        d = np.asarray(d, dtype=np.float64)
        sos = _signal.butter(4, self.input_highpass_hz, btype="high", fs=self.fs, output="sos")
        if self._hp_buf is None:
            self._hp_buf = _signal.sosfilt_zi(sos) * 0.0
        y, self._hp_buf = _signal.sosfilt(sos, d, zi=self._hp_buf)
        return np.asarray(y)

    def _finalize(self) -> None:
        with self._lock:
            if self.state.state == "error":
                # 启动阶段已判错（人声/无周期基频）：保持 error 相位，不产出
                # 误导性的 A/B 报告（此前会在 error 后仍把 phase 置为 done）。
                return
            d = np.concatenate(self.d_buf) if self.d_buf else None
            e = np.concatenate(self.e_buf) if self.e_buf else None
            if d is not None and len(d) > 0:
                self.state.baseline_spl_db = rms_db(d)
            if e is not None and len(e) > 0:
                self.state.cancelling_spl_db = rms_db(e)
                self.state.spl_now_db = rms_db(e[-self.fs:])
            if d is not None and e is not None and len(d) > 0 and len(e) > 0:
                # 稳态段比较（跳过前 30% 收敛期）
                skip = min(int(0.3 * len(e)), self.fs)
                rms_e = np.sqrt(np.mean(e[skip:] ** 2)) if len(e) > skip else np.sqrt(np.mean(e ** 2))
                rms_d = np.sqrt(np.mean(d ** 2))
                if rms_d > 0:
                    self.state.reduction_db = 20.0 * np.log10(max(rms_e, 1e-12) / max(rms_d, 1e-12))
            self.state.phase = "done"

    def _open_stream_guarded(self, cb, open_timeout_s: float = 12.0):
        """带超时保护的 Stream 创建。

        ALSA/PortAudio 在设备被占用时可能无限期阻塞 open；用子线程创建并在
        超时后放弃，避免整条 ANC 线程卡死在打开阶段（此前实测阻塞可达 33s）。
        """
        import sounddevice as sd

        result: list = []
        holder = threading.Event()

        def _open():
            try:
                stream = sd.Stream(
                    samplerate=self.fs, blocksize=self.block,
                    channels=(2, 1) if self.dual_mic else 1,
                    callback=cb, device=(self.in_device, self.out_device),
                    dtype="float32")
                result.append(stream)
            except BaseException as exc:  # noqa: BLE001 —— 子线程内捕获上报
                result.append(exc)
            finally:
                holder.set()

        t = threading.Thread(target=_open, name="anc-stream-open", daemon=True)
        t.start()
        if not holder.wait(open_timeout_s):
            raise RuntimeError(
                "音频流打开超时（%.0fs）：USB 设备可能被占用或未就绪。"
                "请确认没有其他进程正在使用麦克风/扬声器后重试。" % open_timeout_s)
        if len(result) == 0:
            raise RuntimeError("音频流创建失败")
        if isinstance(result[0], BaseException):
            raise result[0]
        return result[0]

    def _run_audio(self) -> None:
        import sounddevice as sd
        from app import capture

        # 输入设备预检：真机 ANC 必须有一支真实误差麦克风
        if self.in_device is None or self.out_device is None:
            duplex = _pick_duplex_default()
            if duplex is not None:
                self.in_device = duplex
                self.out_device = duplex
        if self.in_device is None:
            inp = capture.default_input_device()
            if inp is None:
                raise RuntimeError(
                    "未检测到任何麦克风/输入设备，无法进行真机 ANC。"
                    "请接入 USB 麦克风（或 I2S 编解码器），或使用 --synthetic 自测"
                )
            self.in_device = inp["name"]
        if self.out_device is None:
            self.out_device = self.in_device

        self._pending_cancel = False
        xrun_count = [0]

        def cb(indata, outdata, frames, time_info, status):
            outdata[:] = 0.0
            with self._lock:
                if self.state.state != "running":
                    return
                phase = self.state.phase
            # xrun（欠载/溢出）：丢弃该 block，仅推进相位游标，避免把错乱时序喂给 NLMS
            if status and (status.input_underflow or status.output_underflow
                           or status.input_overflow or status.output_overflow):
                xrun_count[0] += 1
                with self._lock:
                    if self._canceller is not None:
                        self._canceller.skip_block()
                return
            d = indata[:, 0]
            # 双麦模式：第二路 capsule 当风噪参考麦（差分风噪检测用），
            # 不参与 ANC 消除路径（误差麦仍是通道 0）。
            ref = indata[:, 1] if self.dual_mic and indata.shape[1] > 1 else None
            # 输入高通：先去低频风底（对 NLMS 与 f0 都更稳）；风噪门控用未高通的
            # 原始信号判断（高通会破坏"低频占比"这一特征）
            if self.input_highpass_hz > 0 and phase != "baseline":
                d_alg = self._hp_filter(d)
            else:
                d_alg = d
            if phase == "baseline":
                with self._lock:
                    self.d_buf.append(d.copy())
                    self.state.elapsed_s = time.time() - self._start_ts
                    if self.state.elapsed_s >= self.baseline_s:
                        self._pending_cancel = True
            else:
                canceller = self._canceller
                if canceller is not None:
                    # 收集误差麦信号供人声门控（主循环做 FFT 检测，避免占住回调）
                    self._voice_hist.append(d.copy())
                    if len(self._voice_hist) > self._voice_buf_max:
                        del self._voice_hist[: -self._voice_buf_max]
                    # 收集风噪检测缓冲（原始信号；双麦时同时收集参考麦）
                    if self.wind_gate_enabled:
                        if ref is not None:
                            self._wind_hist.append((d.copy(), ref.copy()))
                        else:
                            self._wind_hist.append(d.copy())
                        if len(self._wind_hist) > self._wind_buf_max:
                            del self._wind_hist[: -self._wind_buf_max]
                    if self._voice_mute or (self.wind_gate_enabled and self._wind_mute):
                        # 人声 / 风噪门控：静音反相输出并冻结 NLMS 自适应，
                        # 避免反相波"吃掉"人声 / 对不相关的风喷噪声。
                        canceller.skip_block()
                        outdata[:, 0] = 0.0
                    else:
                        # 反馈中和：从麦克风信号中减去"扬声器回声估计"（延迟后的输出），
                        # 让 NLMS 只跟踪环境噪声，而不是追自己的反相输出。
                        delay_n = canceller.predict_ahead_samples
                        if self.feedback_cancel_gain > 0 and delay_n > 0:
                            echo = self.feedback_cancel_gain * self._read_echo(delay_n)
                            d_alg2 = d_alg - echo
                        else:
                            d_alg2 = d_alg
                        y, _ = canceller.process_block(d_alg2)
                        in_rms = float(np.sqrt(np.mean(d ** 2)))
                        # 延迟扫描期间输出减半：错误相位的试错探针会在误差麦处
                        # 造成短暂 +3dB 爆音，压低增益让校准更安静、读数更干净。
                        out_gain = canceller.output_gain
                        if self._scan_active:
                            out_gain *= 0.5
                        out = compute_safe_output(y, in_rms, out_gain)
                        outdata[:, 0] = out
                        # 输出历史（反馈中和用）：写指针环形缓冲，按时间顺序排布
                        self._out_hist[self._out_pos:self._out_pos + self.block] = out
                        self._out_pos = (self._out_pos + self.block) % len(self._out_hist)
                        spl = None
                        with self._lock:
                            # 残差用误差麦真实采到的信号 d（含扬声器反相波回采），
                            # 而不是数字域理想误差——报告才反映真实声学降噪。
                            self.e_buf.append(d.copy())
                            spl = rms_db(d)
                            self.state.spl_now_db = spl
                            self.state.elapsed_s = time.time() - self._start_ts
                        # 规则式 watchdog（不放锁内，避免与 _watchdog_feed 的加锁冲突）
                        if self.watchdog_enabled and not self._scan_active and spl is not None:
                            self._watchdog_feed(spl)
                        # 延迟自动扫描：试各偏移，取误差麦 RMS 最小者
                        if self._scan_active:
                            self._scan_acc += float(np.mean(d ** 2))
                            self._scan_blocks += 1
                            hold = int(0.4 * self.fs / self.block)
                            if self._scan_blocks >= hold:
                                self._scan_scores.append(self._scan_acc / self._scan_blocks)
                                self._scan_idx += 1
                                if self._scan_idx < len(self._scan_offsets_ms):
                                    canceller.set_predict_ahead_samples(
                                        int(self._scan_offsets_ms[self._scan_idx] * self.fs / 1000.0))
                                    self._scan_acc = 0.0
                                    self._scan_blocks = 0
                                else:
                                    best = int(np.argmin(self._scan_scores))
                                    canceller.set_predict_ahead_samples(
                                        int(self._scan_offsets_ms[best] * self.fs / 1000.0))
                                    self._scan_active = False
                                    with self._lock:
                                        self.state.delay_scan_ms = self._scan_offsets_ms[best]

        stream = self._open_stream_guarded(cb)
        # 以"流真正打开"为基准开始计时：Stream 打开可能因设备占用阻塞数秒，
        # 若以 API 调用时刻计则基线/时长会被墙钟时间污染（基线跳过、立即超时）。
        with self._lock:
            self._start_ts = time.time()
        try:
            with stream:
                while not self._stop.is_set():
                    with self._lock:
                        need_switch = self._pending_cancel
                        if need_switch:
                            self._pending_cancel = False
                    if need_switch:
                        self._begin_cancel()
                    self._voice_gate_tick()
                    self._wind_gate_tick()
                    with self._lock:
                        # 启动阶段判定失败（人声基线 / 无周期基频）→ 结束，不要悬空等 stop
                        if self.state.state == "error":
                            break
                        if self.state.elapsed_s > self.max_duration_s and self.state.phase == "cancelling":
                            break
                    time.sleep(0.05)
        except BaseException:
            import traceback
            traceback.print_exc()
            with self._lock:
                self.state.error = "音频流异常退出（xrun=%d）：%s" % (
                    xrun_count[0], traceback.format_exc(limit=3).splitlines()[-1])
            raise
        with self._lock:
            print("[anc] 流退出: elapsed=%.1fs xrun=%d stop=%s" % (
                self.state.elapsed_s, xrun_count[0], self._stop.is_set()), flush=True)
        self._finalize()

    def _run_synthetic(self) -> None:
        """无硬件自测：用合成打印机噪声模拟误差麦信号，回显增益 echo_gain
        模拟"反相输出被误差麦再次采到"的声学环路。

        双麦合成：参考麦 = 同源打印机噪声 + 独立风噪分量。两 capsule 对同一
        周期声相干，而风噪分量在两路不相关——正好演练双麦差分门控逻辑。
        """
        from app.synth import printer_noise, wind_noise

        n_total = int(self.max_duration_s * self.fs)
        n_baseline = int(self.baseline_s * self.fs)
        noise, _ = printer_noise(fs=self.fs, duration=self.max_duration_s, seed=42)
        noise = noise.astype(np.float64)
        # 第二路参考：同源周期声 + 独立风噪（默认后半段起风，先让门控验证"无风→
        # 有风"的翻转；风幅度取周期声的 60%，足以让差分指标超过门控阈值）
        wind_ref = None
        if self.dual_mic:
            wind = wind_noise(fs=self.fs, duration=self.max_duration_s, seed=7,
                              strength=0.8, cutoff_hz=800.0).astype(np.float64)
            mask = np.zeros(n_total, dtype=np.float64)
            mask[n_total // 2:] = 1.0
            mask[: self.block] = 0.0
            wind_ref = noise + 0.6 * wind * mask
        idx = 0
        y_hist = np.zeros(int(self.fs * 0.05))  # 50ms 回声延迟
        while not self._stop.is_set() and idx < n_total:
            self._voice_gate_tick()
            self._wind_gate_tick()
            blk = noise[idx: idx + self.block]
            if len(blk) < self.block:
                break
            with self._lock:
                phase = self.state.phase
            if phase == "baseline":
                self.d_buf.append(blk.copy())
                if idx + self.block >= n_baseline:
                    self._begin_cancel()
            else:
                canceller = self._canceller
                if canceller is None:
                    break
                if self.wind_gate_enabled:
                    # 双麦合成：参考麦 = 同源周期声 + 独立风噪（后半段起风）
                    if self.dual_mic and wind_ref is not None:
                        self._wind_hist.append((blk.copy(), wind_ref[idx: idx + self.block].copy()))
                    else:
                        self._wind_hist.append(blk.copy())
                    if len(self._wind_hist) > self._wind_buf_max:
                        del self._wind_hist[: -self._wind_buf_max]
                if self._voice_mute or (self.wind_gate_enabled and self._wind_mute):
                    canceller.skip_block()
                    self.e_buf.append(blk.copy())
                    with self._lock:
                        self.state.elapsed_s = idx / self.fs
                        self.state.spl_now_db = rms_db(blk)
                    idx += self.block
                    continue
                desired = blk
                if self.input_highpass_hz > 0:
                    desired = self._hp_filter(blk)
                echo = self.echo_gain * y_hist[: self.block]
                desired = desired + echo
                y, e = canceller.process_block(desired)
                self.e_buf.append(e.copy())
                # 记录输出用于回声（同样走防啸叫整形）
                in_rms = float(np.sqrt(np.mean(desired ** 2)))
                out = compute_safe_output(y, in_rms, canceller.output_gain)
                y_hist = np.roll(y_hist, self.block)
                y_hist[: self.block] = out
            with self._lock:
                self.state.elapsed_s = idx / self.fs
                if self.e_buf:
                    self.state.spl_now_db = rms_db(self.e_buf[-1])
            idx += self.block
        self._finalize()
