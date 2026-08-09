"""Kimi 噪声自动检测 Agent。

把树莓派 ANC 系统的实时/离线音频分析喂给 Kimi K3（OpenAI 兼容 API），通过
tool calling 让模型读取测量数据、比对内置噪声源 Profile、交叉验证规则式结论，
最终输出结构化的噪声源识别与降噪建议。

接入参考 Kimi 开放平台《用 Kimi K3 搭建 Agent》：
- base_url: https://api.moonshot.cn/v1，模型 kimi-k3（可用 KIMI_AGENT_MODEL 覆盖）
- API Key：环境变量 MOONSHOT_API_KEY（或 KIMI_API_KEY），或项目根 .env

未配置 API Key 时 Agent 不可用（configured=False），Web 仪表盘会明确提示，
其余功能不受影响。
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from app.calibration import get_offset_db

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "https://api.moonshot.cn/v1"
MODEL = os.environ.get("KIMI_AGENT_MODEL", "kimi-k3")
REASONING_EFFORT = os.environ.get("KIMI_AGENT_REASONING", "low")  # low | high | max
MAX_TOOL_ROUNDS = 6
DEFAULT_DURATION_S = 3.0
DEFAULT_FS = 48000

DEFAULT_QUESTION = (
    "请检测当前噪声环境：识别噪声源类型与置信度，判断是否值得做主动降噪（ANC），"
    "并给出参考麦克风与静音区建议。"
)


def get_api_key() -> str | None:
    """返回 Kimi API Key。优先环境变量，其次项目根 .env 的 MOONSHOT_API_KEY。"""
    key = os.environ.get("MOONSHOT_API_KEY") or os.environ.get("KIMI_API_KEY")
    if key:
        return key.strip()
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() in ("MOONSHOT_API_KEY", "KIMI_API_KEY"):
                return v.strip().strip("\"'")
    return None


def is_configured() -> bool:
    return config_problem() is None


def config_problem() -> str | None:
    """返回 Agent 不可用的具体原因；可用（已配 key 且 openai 已装）返回 None。

    与 is_configured 分离，便于 UI / API 明确提示「缺 key」还是「缺依赖」——
    用户常已填好 MOONSHOT_API_KEY，但 venv 未装 openai，旧逻辑只会笼统报
    「未配置 MOONSHOT_API_KEY」，误导排查。
    """
    if not get_api_key():
        return "未检测到 MOONSHOT_API_KEY / KIMI_API_KEY（在环境变量或项目根 .env 中设置）"
    try:
        import openai  # noqa: F401
        return None
    except ImportError:
        return "缺少 openai 依赖：请执行 pip install -e \".[agent]\"（Kimi Agent 所需 SDK）"


SYSTEM_PROMPT = """你是部署在树莓派 ANC（主动降噪）系统上的「噪声自动检测 Agent」。
你的任务：读取麦克风采集的音频分析数据，识别当前噪声源类型、评估是否需要主动降噪，
并给出可执行建议。

三个必须区分的概念：
- 环境噪声（noise）：房间/设备产生的环境声源（风机、打印机、空调…），是 ANC 要
  消除的目标，通常是稳态的。
- 声反馈 / 啸叫（howling/squealing）：麦克风→扬声器→麦克风闭环自激振荡。频谱上
  是一个电平随时间增长、频率较稳定的窄带强音调。若 ANC 正在运行且麦克风采到这类
  信号，通常意味着防啸叫余量不足或相位/增益设置不当。它不是"要消除的环境噪声"，
  而是"系统自己发出的声音"，应停止 ANC / 降低增益 / 拉开麦克风与扬声器距离，而不是继续消除。
- 主动降噪（ANC）：用扬声器播放反相波，在误差麦克风处把周期噪声抵消成安静区。
  只对稳态周期噪声（音调）有效；时变宽带噪声（人声、音乐、气动噪声）收益有限。
- 人声/语音（speech/voice）：浊音段虽有周期性基频（约 80–400Hz），但基频随语调
  起伏、帧能量强调制、夹杂大量清音/无声帧，**不是**稳态周期噪声。对说话声启动
  ANC 会反向消除人声（听感上像"吃掉"说话声）。若监控快照或规则分析返回
  is_voice=true（检测到人声），anc_worthwhile 必须为 false，不应建议启动 ANC，
  应建议等人声停止或只对设备噪声降噪。

工作流程：
1. 先用 get_live_noise_snapshot 或 get_latest_analysis 了解当前噪声概况。
2. 调用 check_feedback_or_howling 判断当前信号是「环境噪声」还是「啸叫（声反馈）」，
   并结合 get_anc_status 确认 ANC 引擎是否在运行（区分：环境噪声 → 值得降噪；
   啸叫 → 先处理系统自身问题，不要继续 ANC）。
3. 如需识别噪声源类型，比对该系统内置的噪声源 Profile（3D 打印机、空调、风机、
   硬件工程师风枪、门外扫地机器人等），可调用 run_rule_based_analysis 交叉验证。
4. 综合判断：是否值得做 ANC（安静区直径、扬声器→误差麦延迟、参考麦克风相干性）。
5. **闭环调节（重要）**：若 ANC 引擎正在运行（phase=cancelling），你有权通过
   adjust_anc 直接调节引擎，而不是只给建议：
   - 检测到啸叫/声反馈 → 调用 adjust_anc 降低增益（decrease_gain，步长 0.05–0.1）
     或 set_mic_delay_ms 调整延迟；严重时 stop；
   - 噪声分贝（spl_now_db / reduction_db）没有下降甚至升高 → 调用 adjust_anc
     increase_gain（步长 0.02–0.05）或 set_gain 设置合理增益，继续观察；
   - 每次调节后重新 get_anc_status 确认是否生效，避免过度调节（两次调节之间
     至少间隔几秒让系统稳定）。系统已有规则式 watchdog 兜底，你的调节是其上层补充。
   若 ANC 未在运行（phase=idle/baseline/done），不要调用 adjust_anc。

输出要求：
- 最终回答必须是且仅是一个 JSON 对象，不要包含任何解释文字、代码块或 Markdown。
- JSON 字段（所有文本字段用中文）：
{
  "summary": "一段话总结当前噪声环境",
  "source_id": "已知噪声源 id（3d_printer/ac/fan/heatgun/vacuum_door），无法识别则为 null",
  "source_name": "噪声源中文名，无法识别则为 null",
  "confidence": 0.0到1.0的置信度小数,
  "noise_type": "噪声类型，如 tonal+wideband",
  "dominant_freq_hz": 主频 Hz 或 null,
  "recommended_f0_hz": 建议的 ANC 基频 Hz 或 null,
  "harmonic_family_hz": [谐波家族频率列表],
  "signal_class": "environment_noise | acoustic_feedback | suspected_feedback | uncertain",
  "is_howling": true或false,
  "howling_score": 0.0到1.0,
  "howling_freq_hz": 啸叫/声反馈候选频率 Hz 或 null,
  "anc_worthwhile": true或false,
  "anc_advice": "是否值得降噪及原因（中文，简短）",
  "anc_adjustments": [{"action": "已执行的 adjust_anc 操作", "value": 数值或null, "reason": "调节原因（中文）"}],
  "reference_mic": "参考麦克风摆放建议（中文）",
  "quiet_zone_advice": "静音区/人耳位置建议（中文）",
  "reasons": ["判据1", "判据2"],
  "actions": ["可执行动作1", "动作2"],
  "needs_human": false
}

注意事项：
- 严格基于工具返回的数据，不得编造频谱/分贝数值。
- 若判定为啸叫（is_howling=true）：actions 里必须包含"停止 ANC / 降低输出增益 /
  拉开扬声器与麦克风距离"等处置，anc_worthwhile 应为 false。
- 数据不足（如无麦克风、信号无效）时降低 confidence，并在 reasons 里说明缺口。
- 高频宽带噪声（如风枪气动噪声）ANC 收益有限，应建议被动方案。
- 低频音调（步进电机、压缩机、风扇叶片频率）是 ANC 甜点。
"""

AGENT_TOOLS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "get_live_noise_snapshot",
            "description": "获取当前实时噪声监控快照：SPL、主频、低/中/高频段能量、规则式来源猜测、麦克风状态",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_latest_analysis",
            "description": "获取最近一次完整频谱分析：峰值列表、谐波家族、音调占比、频段能量",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "capture_noise_sample",
            "description": "重新采集一段噪声样本并分析。duration_s 为采样时长（秒）；synthetic=true 时使用合成打印机噪声（无麦克风环境）",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_s": {"type": "number", "description": "采样时长（秒），默认 3"},
                    "synthetic": {"type": "boolean", "description": "是否使用合成噪声，缺省按系统配置"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_noise_profiles",
            "description": "列出系统内置的噪声源 Profile（3D 打印机/空调/风机/风枪/扫地机器人）：噪声类型、频谱特征、ANC 可行性、参考麦克风位置",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_feedback_or_howling",
            "description": "检测当前噪声是否为声反馈/啸叫（麦克风→扬声器→麦克风闭环自激）：采样后分析主峰电平是否随时间增长、音调占比，并关联 ANC 引擎状态。用于区分「环境噪声」与「啸叫」",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_s": {"type": "number", "description": "采样时长（秒），默认 3"},
                    "synthetic": {"type": "boolean", "description": "是否使用合成噪声，缺省按系统配置"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_anc_status",
            "description": "获取实时 ANC 引擎状态：是否运行、相位（基线/降噪中）、基频、输出增益、降噪量。用于判断麦克风采到的信号是否可能来自 ANC 扬声器输出（声反馈场景）",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "adjust_anc",
            "description": "调节正在运行的实时 ANC 引擎参数（增益 / 延迟补偿 / 停止）。用于闭环控制：检测到啸叫/声反馈时调小增益或停止；噪声分贝没有明显下降时适当增大增益。参数均在安全范围限制内生效（增益 0.02–1.0，超出自动截断）",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["set_gain", "increase_gain", "decrease_gain", "set_mic_delay_ms", "stop"],
                        "description": "操作：set_gain 设为指定增益；increase_gain/decrease_gain 相对增减；set_mic_delay_ms 调整扬声器→误差麦延迟补偿；stop 停止 ANC",
                    },
                    "value": {"type": "number", "description": "set_gain 的目标增益（0.02–1.0，超出自动截断）"},
                    "delta": {"type": "number", "description": "increase_gain/decrease_gain 的步长（dB 增益，默认 0.02）"},
                    "mic_delay_ms": {"type": "number", "description": "set_mic_delay_ms 的目标延迟补偿（毫秒）"},
                    "reason": {"type": "string", "description": "调节原因（中文，将写入调节日志）"},
                },
                "required": ["action"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_rule_based_analysis",
            "description": "运行系统内置的规则式噪声源匹配与 ANC 建议，可与你的判断交叉验证",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


def extract_json(text: str | None) -> dict | None:
    """从模型回答中解析 JSON 对象。容忍 markdown 围栏与前后杂散文字。"""
    if not text:
        return None
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(cleaned[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None
    return None


class ToolHandlers:
    """Agent 可用工具的本地实现。monitor 提供实时数据；pre_captured 可注入已采样分析。

    monitor 为 None 时，快照/分析类工具返回错误，Agent 会基于 pre_captured 或
    自身采样继续工作（用于 CLI/测试）。anc_status_provider 提供 ANC 引擎状态
    （用于啸叫/声反馈场景的辅助判读）。
    """

    def __init__(self, monitor: Any = None, fs: int = DEFAULT_FS,
                 pre_captured: dict | None = None,
                 anc_status_provider: Callable[[], dict] | None = None,
                 anc_control: Callable[[str, dict], dict] | None = None,
                 anc_residual_provider: Callable[[], np.ndarray] | None = None) -> None:
        self.monitor = monitor
        self.fs = fs
        self.pre_captured = pre_captured
        self.anc_status_provider = anc_status_provider
        # 可写回调：调整实时 ANC 引擎参数。签名 anc_control(action, params) → dict
        self.anc_control = anc_control
        # 可读回调：返回 ANC 引擎最近残差（误差麦真实信号），供 ANC 运行时的
        # 啸叫检测直接分析，而不是重新采集（运行中麦克风采到的主要是反相波）。
        self.anc_residual_provider = anc_residual_provider
        self._samples: np.ndarray | None = None   # 最近一次采样原始信号（啸叫检测复用）
        self._samples_raw: np.ndarray | None = None  # 原始信号（人声检测用，未剔除瞬态）
        self._samples_synthetic: bool | None = None
        from app.monitor import SYNTHETIC
        self.synth_default = SYNTHETIC

    def call(self, name: str, arguments: dict) -> str:
        handler = getattr(self, name, None)
        if handler is None:
            return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
        try:
            result = handler(**(arguments or {}))
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
        return json.dumps(result, ensure_ascii=False, default=str)

    # ---- 工具实现 ----

    def get_live_noise_snapshot(self) -> dict:
        if self.monitor is None:
            return {"error": "监控线程未接入"}
        return self.monitor.snapshot()

    def get_latest_analysis(self) -> dict:
        if self.pre_captured is not None:
            return {"found": True, "source": "pre_captured", "analysis": self.pre_captured}
        if self.monitor is None:
            return {"found": False, "error": "暂无分析数据"}
        report = self.monitor.last_report()
        if report is None:
            return {"found": False, "error": "暂无分析数据（等待监控采样）"}
        from app.analyze import to_dict
        return {"found": True, "source": "monitor", "analysis": to_dict(report)}

    def capture_noise_sample(self, duration_s: float = DEFAULT_DURATION_S,
                             synthetic: bool | None = None) -> dict:
        samples, use_synth = self._capture_samples(duration_s, synthetic)
        from app.analyze import analyze, to_dict
        report = analyze(samples, self.fs, calibration_offset_db=get_offset_db())
        return {
            "found": True,
            "synthetic": use_synth,
            "duration_s": round(report.duration_s, 2),
            "analysis": to_dict(report),
        }

    def check_feedback_or_howling(self, duration_s: float = DEFAULT_DURATION_S,
                                  synthetic: bool | None = None) -> dict:
        from app.feedback import detect_feedback

        anc = self._anc_snapshot()
        samples, use_synth, source = self._feedback_samples(
            anc, duration_s, synthetic)
        fd = detect_feedback(samples, self.fs)
        # 辅助判读：ANC 正在运行 + 音调占比极高 + 无明显增长 → 疑似已饱和的啸叫
        if (not fd["is_howling"] and fd.get("tonality_ratio") is not None
                and fd["tonality_ratio"] > 0.5 and anc and anc.get("running")):
            fd = dict(fd)
            fd["signal_class"] = "suspected_feedback"
            fd["howling_score"] = max(fd["howling_score"], 0.5)
        return {
            "found": True,
            "synthetic": use_synth,
            "duration_s": round(len(samples) / self.fs, 2),
            "source": source,
            "feedback": fd,
            "anc_engine": anc or {"running": False, "state": "idle", "phase": "idle"},
        }

    def _feedback_samples(self, anc: dict | None, duration_s: float,
                          synthetic: bool | None) -> tuple[np.ndarray, bool, str]:
        """啸叫检测的信号来源。

        ANC 运行时优先分析引擎实时残差（误差麦在降噪中的真实信号），此时麦克风
        主要采到扬声器反相波 + 残余噪声，重新采集没有意义；无残差时回退实时采集。
        """
        if anc and anc.get("running") and self.anc_residual_provider is not None:
            try:
                residual = self.anc_residual_provider()
            except Exception:
                residual = None
            if residual is not None and len(np.asarray(residual)) > 0:
                return np.asarray(residual, dtype=np.float64), False, "anc_residual"
        return self._capture_samples(duration_s, synthetic) + ("live_capture",)

    def get_anc_status(self) -> dict:
        anc = self._anc_snapshot()
        if anc is None:
            return {"running": False, "state": "idle", "phase": "idle",
                    "error": "ANC 引擎未接入"}
        return anc

    def adjust_anc(self, action: str, value: float | None = None,
                   delta: float | None = None, mic_delay_ms: float | None = None,
                   reason: str = "") -> dict:
        """调节实时 ANC 引擎（增益 / 延迟 / 停止）。

        action ∈ set_gain | increase_gain | decrease_gain | set_mic_delay_ms | stop。
        未接入可写回调时返回错误（Agent 仍可输出建议）。
        """
        if self.anc_control is None:
            return {"ok": False, "error": "ANC 引擎可写控制未接入（仅检测模式）"}
        params: dict[str, Any] = {}
        if value is not None:
            params["value"] = value
        if delta is not None:
            params["delta"] = delta
        if mic_delay_ms is not None:
            params["mic_delay_ms"] = mic_delay_ms
        params["reason"] = reason or ""
        try:
            result = self.anc_control(action, params)
            if not isinstance(result, dict):
                return {"ok": False, "error": f"ANC 控制返回类型错误: {type(result).__name__}"}
            return result
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def list_noise_profiles(self) -> dict:
        from app.source_id import load_profiles
        profiles = load_profiles()
        return {"profiles": [
            {
                "id": p.get("id"),
                "name": p.get("name"),
                "noise_type": p.get("noise_type"),
                "signatures": p.get("signatures", []),
                "anc_feasibility": p.get("anc_feasibility"),
                "reference_mic": p.get("reference_mic"),
                "notes": p.get("notes"),
            } for p in profiles
        ]}

    def run_rule_based_analysis(self) -> dict:
        if self.pre_captured is not None:
            report = _report_from_dict(self.pre_captured)
        elif self.monitor is not None and self.monitor.last_report() is not None:
            report = self.monitor.last_report()
        else:
            return {"found": False, "error": "暂无分析数据"}
        from app.source_id import match_sources, recommend_anc
        hits = match_sources(report)
        rec = recommend_anc(report, voice=self._voice_context())
        return {
            "found": True,
            "matches": [h.__dict__ for h in hits[:3]],
            "recommendation": rec,
        }

    def _voice_context(self) -> dict | None:
        """人声判定上下文：优先用最近一次采样的原始信号，否则取监控状态。"""
        if self._samples_raw is not None and len(self._samples_raw) > 0:
            from app.voice import detect_voice
            return detect_voice(self._samples_raw, self.fs)
        if self.monitor is not None and self.monitor.state.is_voice is not None:
            return {
                "is_voice": self.monitor.state.is_voice,
                "score": self.monitor.state.voice_score,
                "reasons": list(self.monitor.state.voice_reasons),
            }
        return None

    # ---- 内部 ----

    def _capture_samples(self, duration_s: float,
                         synthetic: bool | None) -> tuple[np.ndarray, bool]:
        """采集原始样本并缓存（啸叫检测复用）。返回 (samples, 实际是否合成)。"""
        use_synth = self.synth_default if synthetic is None else synthetic
        if use_synth:
            import time as _t

            from app.synth import printer_noise
            samples, _ = printer_noise(fs=self.fs, duration=duration_s, seed=int(_t.time()))
            self._samples_raw = samples
        else:
            from app import capture
            from app.analyze import stable_segment
            raw = capture.record_buffer(duration_s, fs=self.fs)
            self._samples_raw = raw
            samples = stable_segment(raw, self.fs)
            if len(samples) == 0:
                raise RuntimeError("无有效信号（瞬态过多）")
        samples = np.asarray(samples, dtype=np.float64)
        self._samples = samples
        self._samples_synthetic = use_synth
        return samples, use_synth

    def _anc_snapshot(self) -> dict | None:
        if self.anc_status_provider is None:
            return None
        st = self.anc_status_provider()
        if not isinstance(st, dict):
            return None
        running = st.get("state") == "running" or st.get("phase") in ("baseline", "cancelling")
        return {**st, "running": running}


def _report_from_dict(d: dict) -> Any:
    """把 to_dict() 输出的字典还原成 AnalysisReport（供规则式分析交叉验证）。

    仅用于 pre_captured 场景；AnalysisReport 字段与 to_dict 输出一一对应。
    """
    from dataclasses import fields

    from app.analyze import AnalysisReport, TonalPeak
    values = {}
    for f in fields(AnalysisReport):
        if f.name not in d:
            continue
        v = d[f.name]
        if f.name == "peaks" and isinstance(v, list):
            v = [TonalPeak(freq=p["freq"], level_db=p["level_db"],
                           prominence_db=p["prominence_db"],
                           harmonic_order=p.get("harmonic_order", 1)) for p in v]
        values[f.name] = v
    return AnalysisReport(**values)


class NoiseDetectionAgent:
    """Kimi K3 Agent：工具循环 + 最终 JSON 结论。client 可注入（测试用）。

    # 工具循环
    #
    #   user question ─► Kimi K3 ─► message.tool_calls?
    #     ├─ 是 ─► 本地 handler 执行 ─► tool 结果回填消息 ─► 回到 Kimi（≤6 轮）
    #     └─ 否 ─► 解析最终 JSON ─► done（parse_error 标记解析失败）
    #
    # 闭环调节（ANC 运行时）：
    #   啸叫/声反馈 ─► decrease_gain / set_mic_delay_ms / stop
    #   降噪不足   ─► increase_gain / set_gain
    #   每次调节后重查 get_anc_status 确认生效，避免过度调节。
    #   watchdog 是规则式兜底（亚秒级），Agent 调节是其上层补充。
    """

    def __init__(self, api_key: str, model: str | None = None,
                 base_url: str = BASE_URL, reasoning_effort: str | None = REASONING_EFFORT,
                 max_tool_rounds: int = MAX_TOOL_ROUNDS,
                 client: Any = None) -> None:
        if client is None:
            from openai import OpenAI
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=90.0)
        self.client = client
        self.model = model or MODEL
        self.reasoning_effort = reasoning_effort
        self.max_tool_rounds = max_tool_rounds

    def run(self, handlers: ToolHandlers, question: str = DEFAULT_QUESTION) -> dict:
        messages: list[Any] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        tools_used: list[str] = []
        rounds = 0
        for _ in range(self.max_tool_rounds):
            rounds += 1
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": messages,
                "tools": AGENT_TOOLS,
                "max_completion_tokens": 4096,
            }
            if self.reasoning_effort:
                kwargs["reasoning_effort"] = self.reasoning_effort
            response = self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]
            message = choice.message
            # 追加完整 assistant message，保留 reasoning_content / tool_calls 上下文
            messages.append(message)

            if not message.tool_calls:
                if not message.content:
                    raise RuntimeError("模型未返回任何内容")
                return self._finalize(message.content, rounds, tools_used)

            for tc in message.tool_calls:
                name = tc.function.name
                try:
                    arguments = json.loads(tc.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("工具参数必须是 JSON 对象")
                except Exception as exc:
                    arguments = {}
                    result = json.dumps({"error": f"{type(exc).__name__}: {exc}"},
                                         ensure_ascii=False)
                else:
                    tools_used.append(name)
                    result = handlers.call(name, arguments)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        raise RuntimeError(f"工具调用超过最大轮数 {self.max_tool_rounds}")

    def _finalize(self, answer: str, rounds: int, tools_used: list[str]) -> dict:
        parsed = extract_json(answer)
        return {
            "rounds": rounds,
            "tools_used": tools_used,
            "raw_answer": answer,
            "result": parsed,
            "parse_error": None if parsed else "模型输出无法解析为 JSON",
        }


@dataclass
class AgentStatus:
    state: str = "idle"            # idle | running | done | error
    configured: bool = False
    configured_reason: str = ""    # 不可用原因（缺 key / 缺依赖），可用时为空
    model: str = MODEL
    base_url: str = BASE_URL
    message: str = ""
    started_at: float | None = None
    finished_at: float | None = None
    elapsed_s: float | None = None
    rounds: int = 0
    tools_used: list[str] = field(default_factory=list)
    result: dict | None = None
    raw_answer: str | None = None
    error: str | None = None


class NoiseAgentWorker:
    """后台执行 Kimi 噪声检测 Agent。同一时刻只允许一个任务。

    # 状态机
    #
    #   idle ──start()──► running ──_run 成功──► done
    #     │                  │                   │
    #     │                  └──_run 异常──────► error
    #     │
    #     ├─ 未配置 / 已在运行 ──► 拒绝（started=False + 原因）
    #
    # 降级路径：fresh_sample 重新采样失败（无麦克风）→ 用监控线程最近数据继续，
    # 结果 message 里附加说明。监控线程暂停/恢复由 finally 保证，异常也不泄漏。
    """

    def __init__(self, monitor: Any = None,
                 anc_status_provider: Callable[[], dict] | None = None,
                 anc_control: Callable[[str, dict], dict] | None = None,
                 anc_residual_provider: Callable[[], np.ndarray] | None = None) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.monitor = monitor
        self.anc_status_provider = anc_status_provider
        self.anc_control = anc_control
        self.anc_residual_provider = anc_residual_provider
        self.status = AgentStatus(configured=is_configured(),
                                  configured_reason=config_problem() or "")

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status_dict(self) -> dict:
        with self._lock:
            return asdict(self.status)

    def start(self, fresh_sample: bool = True, duration_s: float = DEFAULT_DURATION_S,
              synthetic: bool | None = None,
              question: str | None = None) -> dict:
        if self.is_running():
            return {"started": False, "message": "Agent 已在运行，请等待完成"}
        if not is_configured():
            return {"started": False, "message": config_problem()}
        with self._lock:
            self.status = AgentStatus(
                state="running", configured=True, model=MODEL, base_url=BASE_URL,
                message="Agent 启动", started_at=time.time(),
            )
        self._thread = threading.Thread(
            target=self._run, name="kimi-agent", daemon=True,
            args=(fresh_sample, duration_s, synthetic, question),
        )
        self._thread.start()
        return {"started": True}

    # ---- 内部 ----

    def _run(self, fresh_sample: bool, duration_s: float, synthetic: bool | None,
             question: str | None) -> None:
        paused_by_us = False
        fresh_note = ""
        try:
            if self.monitor is not None:
                # Agent 可能需要重新采样，暂停监控线程避免抢占音频设备
                if not self.monitor.state.paused:
                    self.monitor.set_paused(True)
                    paused_by_us = True
                # 等监控释放常驻录音流后再采样，避免设备被占用
                self.monitor.wait_paused()

            pre_captured = None
            handlers = ToolHandlers(monitor=self.monitor,
                                    anc_status_provider=self.anc_status_provider,
                                    anc_control=self.anc_control,
                                    anc_residual_provider=self.anc_residual_provider)
            if fresh_sample:
                try:
                    pre_captured = handlers.capture_noise_sample(
                        duration_s=duration_s, synthetic=synthetic)["analysis"]
                except Exception as exc:
                    # 采样失败（无麦克风等）：降级用监控已有数据继续分析
                    fresh_note = f"重新采样失败（{exc}），已改用最近一次监控数据"
            if pre_captured is not None:
                handlers.pre_captured = pre_captured

            agent = NoiseDetectionAgent(api_key=get_api_key())
            outcome = agent.run(handlers, question=question or DEFAULT_QUESTION)

            with self._lock:
                st = self.status
                st.state = "done"
                st.finished_at = time.time()
                st.elapsed_s = round(st.finished_at - (st.started_at or st.finished_at), 1)
                st.rounds = outcome["rounds"]
                st.tools_used = outcome["tools_used"]
                st.result = outcome["result"]
                st.raw_answer = outcome["raw_answer"]
                st.message = "检测完成" if outcome["result"] else "检测完成（输出解析失败）"
                if fresh_note:
                    st.message += f"。{fresh_note}"
        except Exception as exc:
            with self._lock:
                st = self.status
                st.state = "error"
                st.finished_at = time.time()
                st.elapsed_s = round(st.finished_at - (st.started_at or st.finished_at), 1)
                st.error = str(exc)
                st.message = "检测失败"
        finally:
            if paused_by_us and self.monitor is not None:
                self.monitor.set_paused(False)
