"""实时 ANC 引擎：block 级自适应谐波消除（自参考，无需参考麦克风）。

M2 现场 demo 的核心：用一支误差麦克风（USB 麦）采噪声，在树莓派上实时
估计基频并对各谐波做 NLMS 幅度/相位跟踪，输出反相波形到扬声器，在误差麦处
形成安静区。对步进电机 / 风扇叶片等周期噪声有效。

- `BlockHarmonicCanceller`：向量化 block 处理（替代逐样本 step，48 kHz 实时可跑）。
- `LiveANCEngine`：sounddevice 双工流线程，分两段采集（ANC off 基线 / ANC on 残差），
  线程安全 `status()` 供仪表盘轮询；`synthetic=True` 时无硬件自测。
- 防啸叫：低输出增益 + tanh + 硬限幅 + 输出/输入 RMS 比例门控 + 权重范数上限 +
  xrun 时丢弃 block 仅推进相位游标（避免把错乱时序喂给 NLMS）。
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.analyze import rms_db


class BlockHarmonicCanceller:
    """向量化自适应谐波消除。每个谐波一组 cos/sin 权重，block NLMS 更新。"""

    def __init__(self, fs: float, f0: float, max_harmonics: int = 10,
                 mu: float = 1e-3, block: int = 512, output_gain: float = 0.08,
                 max_weight_norm: float = 5.0,
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
                        clip_level: float = 0.12, ratio: float = 4.0,
                        quiet_rms: float = 1e-3) -> np.ndarray:
    """反相输出防啸叫整形。

    - 输入能量极低（安静）时输出静音，避免向安静环境喷出自身噪声。
    - tanh 压缩后硬限幅到 ±clip_level：即使权重发散也不能把扬声器炸响。
    - 输出 RMS 不得超过 ratio × 输入 RMS（正反馈自激时按比例缩回）。
    """
    if in_rms < quiet_rms:
        return np.zeros_like(y)
    out = np.clip(-gain * np.tanh(y), -clip_level, clip_level)
    out_rms = float(np.sqrt(np.mean(out ** 2)))
    max_allowed = ratio * max(in_rms, 1e-12)
    if out_rms > max_allowed:
        out *= max_allowed / out_rms
    return out


@dataclass
class LiveState:
    state: str = "idle"  # idle | running | stopping | stopped | error
    phase: str = "idle"  # idle | baseline | cancelling | done
    f0: float | None = None
    gain: float = 0.08
    mic_delay_ms: float = 5.0  # 扬声器→误差麦延迟补偿（当前生效值）
    spl_now_db: float | None = None
    baseline_spl_db: float | None = None
    cancelling_spl_db: float | None = None
    reduction_db: float | None = None
    elapsed_s: float = 0.0
    error: str | None = None


class LiveANCEngine:
    """实时 ANC 环路线程。无参考麦克风：从误差信号自参考估计并消除谐波。"""

    def __init__(self, fs: int = 48000, in_device=None, out_device=None,
                 block: int = 512, f0: float | None = None, max_harmonics: int = 10,
                 mu: float = 2e-2, output_gain: float = 0.08, baseline_s: float = 5.0,
                 max_duration_s: float = 60.0, synthetic: bool = False,
                 echo_gain: float = 0.0,
                 speaker_mic_delay_s: float = 0.005) -> None:
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

        self.state = LiveState()
        self.d_buf: list[np.ndarray] = []   # 基线（ANC off）
        self.e_buf: list[np.ndarray] = []   # 残差（ANC on）
        self._canceller: BlockHarmonicCanceller | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_ts = 0.0

    # ---- 控制 ----

    def start(self) -> dict:
        if self._thread and self._thread.is_alive():
            return {"started": False, "message": "已在运行"}
        with self._lock:
            self.state = LiveState(state="running", phase="baseline",
                                   gain=self.output_gain, f0=self.f0_init,
                                   mic_delay_ms=self.speaker_mic_delay_s * 1000.0)
            self.d_buf = []
            self.e_buf = []
            self._canceller = None
            self._start_ts = time.time()
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
            elapsed = round(st.elapsed_s, 1)
            return {
                "state": st.state,
                "phase": st.phase,
                "f0": st.f0,
                "gain": st.gain,
                "mic_delay_ms": round(st.mic_delay_ms, 1),
                "spl_now_db": round(st.spl_now_db, 1) if st.spl_now_db is not None else None,
                "baseline_spl_db": round(st.baseline_spl_db, 1) if st.baseline_spl_db is not None else None,
                "cancelling_spl_db": round(st.cancelling_spl_db, 1) if st.cancelling_spl_db is not None else None,
                "reduction_db": round(st.reduction_db, 1) if st.reduction_db is not None else None,
                "elapsed_s": elapsed,
                "synthetic": self.synthetic,
                "error": st.error,
            }

    def get_signals(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """返回拼接后的 (baseline d, 残差 e)，供 A/B 评估。"""
        with self._lock:
            d = np.concatenate(self.d_buf) if self.d_buf else None
            e = np.concatenate(self.e_buf) if self.e_buf else None
            return d, e

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
            buf = self.d_buf
        if not buf:
            return
        d = np.concatenate(buf)
        with self._lock:
            self.state.baseline_spl_db = rms_db(d)
        # 只用尾部 ~1s 估基频：自相关是 O(n²)，整段 3s/48k 会阻塞主循环数秒
        tail = d[-int(self.fs):] if len(d) >= self.fs else d
        f0 = self.f0_init or estimate_fundamental(tail, self.fs)
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

    def _finalize(self) -> None:
        with self._lock:
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

    def _run_audio(self) -> None:
        import sounddevice as sd

        self._pending_cancel = False

        def cb(indata, outdata, frames, time_info, status):
            outdata[:] = 0.0
            with self._lock:
                if self.state.state != "running":
                    return
                phase = self.state.phase
            # xrun（欠载/溢出）：丢弃该 block，仅推进相位游标，避免把错乱时序喂给 NLMS
            if status and (status.input_underflow or status.output_underflow
                           or status.input_overflow or status.output_overflow):
                with self._lock:
                    if self._canceller is not None:
                        self._canceller.skip_block()
                return
            d = indata[:, 0]
            if phase == "baseline":
                with self._lock:
                    self.d_buf.append(d.copy())
                    self.state.elapsed_s = time.time() - self._start_ts
                    if self.state.elapsed_s >= self.baseline_s:
                        self._pending_cancel = True
            else:
                canceller = self._canceller
                if canceller is not None:
                    y, e = canceller.process_block(d)
                    in_rms = float(np.sqrt(np.mean(d ** 2)))
                    outdata[:, 0] = compute_safe_output(
                        y, in_rms, canceller.output_gain)
                    with self._lock:
                        self.e_buf.append(e.copy())
                        self.state.spl_now_db = rms_db(e)
                        self.state.elapsed_s = time.time() - self._start_ts

        stream = sd.Stream(samplerate=self.fs, blocksize=self.block, channels=1,
                           callback=cb, device=(self.in_device, self.out_device),
                           dtype="float32")
        with stream:
            while not self._stop.is_set():
                with self._lock:
                    need_switch = self._pending_cancel
                    if need_switch:
                        self._pending_cancel = False
                if need_switch:
                    self._begin_cancel()
                with self._lock:
                    if self.state.elapsed_s > self.max_duration_s and self.state.phase == "cancelling":
                        break
                time.sleep(0.05)
        self._finalize()

    def _run_synthetic(self) -> None:
        """无硬件自测：用合成打印机噪声模拟误差麦信号，回显增益 echo_gain
        模拟"反相输出被误差麦再次采到"的声学环路。"""
        from app.synth import printer_noise

        n_total = int(self.max_duration_s * self.fs)
        n_baseline = int(self.baseline_s * self.fs)
        noise, _ = printer_noise(fs=self.fs, duration=self.max_duration_s, seed=42)
        noise = noise.astype(np.float64)
        idx = 0
        y_hist = np.zeros(int(self.fs * 0.05))  # 50ms 回声延迟
        while not self._stop.is_set() and idx < n_total:
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
                echo = self.echo_gain * y_hist[: self.block]
                desired = blk + echo
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
