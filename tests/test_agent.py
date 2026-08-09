"""Kimi 噪声检测 Agent 的离线测试（不发起真实网络请求）。"""
from __future__ import annotations

import json
import time
import types

import numpy as np
import pytest

from app.agent import (
    NoiseAgentWorker,
    NoiseDetectionAgent,
    ToolHandlers,
    config_problem,
    extract_json,
)
from app.analyze import analyze, to_dict
from app.synth import printer_noise


# ---------- extract_json ----------

def test_extract_json_plain():
    assert extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_with_markdown_fence():
    text = '```json\n{"source_id": "fan"}\n```'
    assert extract_json(text) == {"source_id": "fan"}


def test_extract_json_with_wrapping_text():
    text = '好的，结论如下：\n{"summary": "测试"}\n以上。'
    assert extract_json(text) == {"summary": "测试"}


def test_extract_json_invalid():
    assert extract_json("完全不是 JSON") is None
    assert extract_json(None) is None


# ---------- ToolHandlers（纯本地，无网络） ----------

def _synthetic_report_dict(duration: float = 3.0) -> dict:
    x, _ = printer_noise(fs=16000, duration=duration, seed=7)
    return to_dict(analyze(x, 16000))


def test_tool_capture_noise_sample_synthetic():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("capture_noise_sample",
                                   {"duration_s": 3.0, "synthetic": True}))
    assert res["found"] is True
    assert res["synthetic"] is True
    assert res["analysis"]["dominant_freq"] is not None


def test_tool_list_noise_profiles():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("list_noise_profiles", {}))
    ids = [p["id"] for p in res["profiles"]]
    assert "3d_printer" in ids
    assert "ac" in ids


def test_tool_rule_based_analysis_with_pre_captured():
    handlers = ToolHandlers(pre_captured=_synthetic_report_dict())
    res = json.loads(handlers.call("run_rule_based_analysis", {}))
    assert res["found"] is True
    assert res["matches"], "合成打印机噪声应命中规则式来源匹配"


def test_tool_get_latest_analysis_prefers_pre_captured():
    handlers = ToolHandlers(pre_captured={"dominant_freq": 123.0})
    res = json.loads(handlers.call("get_latest_analysis", {}))
    assert res["source"] == "pre_captured"


def test_tool_unknown_name_returns_error():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("no_such_tool", {}))
    assert "error" in res


def test_tool_check_feedback_or_howling_synthetic_noise():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("check_feedback_or_howling",
                                   {"duration_s": 3.0, "synthetic": True}))
    assert res["found"] is True
    assert res["feedback"]["signal_class"] == "environment_noise"
    assert res["anc_engine"]["running"] is False


def test_tool_check_feedback_reuses_cached_samples():
    handlers = ToolHandlers()
    handlers.call("capture_noise_sample", {"duration_s": 3.0, "synthetic": True})
    res = json.loads(handlers.call("check_feedback_or_howling", {"synthetic": True}))
    assert res["found"] is True
    assert res["synthetic"] is True


def test_tool_get_anc_status_no_provider():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("get_anc_status", {}))
    assert res["running"] is False
    assert "error" in res


def test_tool_get_anc_status_with_provider():
    def provider():
        return {"state": "running", "phase": "cancelling", "f0": 120.0,
                "gain": 0.12, "reduction_db": -8.5}
    handlers = ToolHandlers(anc_status_provider=provider)
    res = json.loads(handlers.call("get_anc_status", {}))
    assert res["running"] is True
    assert res["phase"] == "cancelling"


def test_tool_adjust_anc_with_control_callback():
    """Agent 可写：adjust_anc 应把 action/参数透传给 anc_control 回调。"""
    calls = []

    def control(action, params):
        calls.append((action, params))
        return {"ok": True, "gain": 0.2, "action": action}

    handlers = ToolHandlers(anc_control=control)
    res = json.loads(handlers.call("adjust_anc",
                                   {"action": "set_gain", "value": 0.2, "reason": "测试"}))
    assert res["ok"] is True
    assert calls == [("set_gain", {"value": 0.2, "reason": "测试"})]


def test_tool_adjust_anc_without_callback_returns_error():
    handlers = ToolHandlers()
    res = json.loads(handlers.call("adjust_anc", {"action": "increase_gain"}))
    assert res["ok"] is False
    assert "未接入" in res["error"]


def test_tool_adjust_anc_callback_exception_surfaced():
    def control(action, params):
        raise ValueError("boom")
    handlers = ToolHandlers(anc_control=control)
    res = json.loads(handlers.call("adjust_anc", {"action": "stop"}))
    assert res["ok"] is False
    assert "boom" in res["error"]


def test_tool_check_feedback_suspected_when_anc_running_and_tonal():
    """ANC 运行 + 高音调稳态信号 → 应标记为疑似啸叫（suspected_feedback）。"""
    fs = 16000
    t = np.arange(int(fs * 3.0)) / fs
    tonal = 0.1 * np.sin(2.0 * np.pi * 120.0 * t)  # 恒定低频纯音（高音调、无增长）

    def provider():
        return {"state": "running", "phase": "cancelling", "f0": 120.0, "gain": 0.12}

    class FakeHandlers(ToolHandlers):
        def _capture_samples(self, duration_s, synthetic):
            return tonal, False

    handlers = FakeHandlers(anc_status_provider=provider)
    res = json.loads(handlers.call("check_feedback_or_howling", {}))
    assert res["feedback"]["signal_class"] == "suspected_feedback"
    assert res["anc_engine"]["running"] is True


# ---------- Agent 工具循环（stub client） ----------

class FakeMsg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"
        self.reasoning_content = None


class FakeToolCall:
    def __init__(self, name, arguments="{}"):
        self.id = "call_1"
        self.function = types.SimpleNamespace(name=name, arguments=arguments)


class FakeCompletions:
    def __init__(self, responses):
        self.responses = list(responses)

    def create(self, **kwargs):
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=self.responses.pop(0))]
        )


class FakeClient:
    def __init__(self, responses):
        self.chat = types.SimpleNamespace(completions=FakeCompletions(responses))


def test_agent_direct_json_answer():
    answer = json.dumps({"source_id": "fan", "anc_worthwhile": False, "confidence": 0.9},
                        ensure_ascii=False)
    client = FakeClient([FakeMsg(content=answer)])
    agent = NoiseDetectionAgent(api_key="test-key", client=client)
    outcome = agent.run(ToolHandlers())
    assert outcome["result"]["source_id"] == "fan"
    assert outcome["rounds"] == 1
    assert outcome["tools_used"] == []


def test_agent_tool_round_then_answer():
    answer = json.dumps({"source_id": "3d_printer", "anc_worthwhile": True}, ensure_ascii=False)
    client = FakeClient([
        FakeMsg(tool_calls=[FakeToolCall("list_noise_profiles")]),
        FakeMsg(content=answer),
    ])
    agent = NoiseDetectionAgent(api_key="test-key", client=client)
    outcome = agent.run(ToolHandlers())
    assert outcome["rounds"] == 2
    assert outcome["tools_used"] == ["list_noise_profiles"]
    assert outcome["result"]["source_id"] == "3d_printer"


def test_agent_loop_handles_unknown_tool_error():
    answer = json.dumps({"summary": "ok"}, ensure_ascii=False)
    client = FakeClient([
        FakeMsg(tool_calls=[FakeToolCall("not_a_real_tool")]),
        FakeMsg(content=answer),
    ])
    agent = NoiseDetectionAgent(api_key="test-key", client=client)
    outcome = agent.run(ToolHandlers())
    assert outcome["result"]["summary"] == "ok"


def test_agent_max_rounds_guard():
    client = FakeClient([FakeMsg(tool_calls=[FakeToolCall("list_noise_profiles")])] * 4)
    agent = NoiseDetectionAgent(api_key="test-key", client=client, max_tool_rounds=2)
    with pytest.raises(RuntimeError, match="最大轮数"):
        agent.run(ToolHandlers())


# ---------- Worker ----------

def test_worker_unconfigured(monkeypatch):
    monkeypatch.setattr("app.agent.is_configured", lambda: False)
    monkeypatch.setattr("app.agent.get_api_key", lambda: None)
    w = NoiseAgentWorker()
    assert w.status.configured is False
    r = w.start()
    assert r["started"] is False
    assert "MOONSHOT_API_KEY" in r["message"]


# ---- 配置诊断（Bug: 用户已填 key 但 Agent 仍启动不了） ----

def test_config_problem_missing_key(monkeypatch):
    """没 key → 明确提示缺 key。"""
    monkeypatch.setattr("app.agent.get_api_key", lambda: None)
    assert config_problem() is not None
    assert "MOONSHOT_API_KEY" in config_problem()


def test_config_problem_missing_openai_dep(monkeypatch):
    """key 已配置但 openai 未安装 → 提示缺依赖而非缺 key。

    复现用户反馈的「API key 我都提供了，但 AI 启动不了」：
    get_api_key 正常返回，但 venv 没装 openai，is_configured 仍为 False。
    """
    import builtins

    monkeypatch.setattr("app.agent.get_api_key", lambda: "sk-test-key")
    _orig_import = builtins.__import__

    def _no_openai(name, *args, **kwargs):
        if name.split(".")[0] == "openai":
            raise ImportError("No module named 'openai'")
        return _orig_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_openai)
    assert config_problem() is not None
    assert "openai" in config_problem()
    assert "MOONSHOT_API_KEY" not in config_problem()


def test_worker_unconfigured_reports_missing_dep(monkeypatch):
    """start() 被拒时 message 说明真实原因（缺 openai 依赖），而非笼统缺 key。"""
    monkeypatch.setattr("app.agent.get_api_key", lambda: "sk-test-key")

    def _no_openai(name, *args, **kwargs):
        if name in ("openai",):
            raise ImportError("No module named 'openai'")
        return _orig_import(name, *args, **kwargs)

    import builtins
    _orig_import = builtins.__import__
    monkeypatch.setattr(builtins, "__import__", _no_openai)
    w = NoiseAgentWorker()
    r = w.start()
    assert r["started"] is False
    assert "openai" in r["message"]
    assert w.status.configured is False
    assert "openai" in (w.status.configured_reason or "")


# ---- Worker 状态机（stub client 端到端跑 _run，无真实网络） ----

class StubAgent:
    """成功路径：run() 返回可解析结论。记录收到的 handlers 供断言。"""

    instances = []

    def __init__(self, api_key, **kwargs):
        self.api_key = api_key
        self.handlers = None
        StubAgent.instances.append(self)

    def run(self, handlers, question=None):
        self.handlers = handlers
        self.question = question
        return {"rounds": 2, "tools_used": ["get_anc_status"],
                "raw_answer": '{"summary": "ok", "source_id": "fan"}',
                "result": {"summary": "ok", "source_id": "fan"}}


class UnparsableAgent(StubAgent):
    def run(self, handlers, question=None):
        return {"rounds": 1, "tools_used": [], "raw_answer": "不是 JSON",
                "result": None}


class RaisingAgent(StubAgent):
    def run(self, handlers, question=None):
        raise RuntimeError("kimi api boom")


class SlowAgent(StubAgent):
    def run(self, handlers, question=None):
        time.sleep(0.2)
        return {"rounds": 1, "tools_used": [], "raw_answer": "{}", "result": {}}


class FakeMonitor:
    def __init__(self):
        self.state = types.SimpleNamespace(paused=False)
        self.pause_calls = []

    def set_paused(self, v):
        self.pause_calls.append(v)
        self.state.paused = v


def _ready_worker(monkeypatch, agent_cls=StubAgent, monitor=None):
    monkeypatch.setattr("app.agent.is_configured", lambda: True)
    monkeypatch.setattr("app.agent.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.agent.NoiseDetectionAgent", agent_cls)
    StubAgent.instances = []
    return NoiseAgentWorker(monitor=monitor)


def test_worker_runs_to_done(monkeypatch):
    w = _ready_worker(monkeypatch)
    r = w.start(synthetic=True)
    assert r["started"] is True
    w._thread.join(timeout=10.0)
    st = w.status_dict()
    assert st["state"] == "done"
    assert st["result"] == {"summary": "ok", "source_id": "fan"}
    assert st["rounds"] == 2
    assert st["tools_used"] == ["get_anc_status"]
    assert st["error"] is None
    assert st["message"].startswith("检测完成")
    assert not w.is_running()
    # 采样得到的 pre_captured 应注入 handlers（Agent 能读到分析数据）
    assert StubAgent.instances[0].handlers.pre_captured is not None


def test_worker_done_with_unparsable_output(monkeypatch):
    w = _ready_worker(monkeypatch, agent_cls=UnparsableAgent)
    w.start(synthetic=True)
    w._thread.join(timeout=10.0)
    st = w.status_dict()
    assert st["state"] == "done"
    assert st["result"] is None
    assert "解析失败" in st["message"]


def test_worker_error_state(monkeypatch):
    w = _ready_worker(monkeypatch, agent_cls=RaisingAgent)
    w.start(synthetic=True)
    w._thread.join(timeout=10.0)
    st = w.status_dict()
    assert st["state"] == "error"
    assert "kimi api boom" in st["error"]
    assert st["message"] == "检测失败"


def test_worker_fresh_sample_failure_degrades(monkeypatch):
    """采样失败（无麦克风）→ 降级用监控数据继续，message 附加说明。"""
    w = _ready_worker(monkeypatch)

    def boom(self, **kwargs):
        raise RuntimeError("无有效信号")

    monkeypatch.setattr("app.agent.ToolHandlers.capture_noise_sample", boom)
    w.start(synthetic=True)
    w._thread.join(timeout=10.0)
    st = w.status_dict()
    assert st["state"] == "done"
    assert "重新采样失败" in st["message"]
    assert st["result"] is not None
    assert StubAgent.instances[0].handlers.pre_captured is None


def test_worker_skips_fresh_sample_when_false(monkeypatch):
    w = _ready_worker(monkeypatch)
    calls = []

    def fake_capture(self, **kwargs):
        calls.append(1)
        return {}

    monkeypatch.setattr("app.agent.ToolHandlers.capture_noise_sample", fake_capture)
    w.start(fresh_sample=False, synthetic=True)
    w._thread.join(timeout=10.0)
    assert calls == [], "fresh_sample=False 不应触发重新采样"
    assert StubAgent.instances[0].handlers.pre_captured is None


def test_worker_pauses_and_resumes_monitor(monkeypatch):
    mon = FakeMonitor()
    w = _ready_worker(monkeypatch, monitor=mon)
    w.start(synthetic=True)
    w._thread.join(timeout=10.0)
    assert mon.pause_calls == [True, False]
    assert mon.state.paused is False


def test_worker_does_not_resume_already_paused_monitor(monkeypatch):
    mon = FakeMonitor()
    mon.set_paused(True)  # 已被其他流程暂停（如 ANC 正在运行）
    mon.pause_calls.clear()  # 清掉 setup 调用，只统计 worker 的
    w = _ready_worker(monkeypatch, monitor=mon)
    w.start(synthetic=True)
    w._thread.join(timeout=10.0)
    assert mon.pause_calls == [], "不是我们暂停的，不应恢复"
    assert mon.state.paused is True


def test_worker_rejects_concurrent_start(monkeypatch):
    w = _ready_worker(monkeypatch, agent_cls=SlowAgent)
    r1 = w.start(synthetic=True)
    assert r1["started"] is True
    r2 = w.start(synthetic=True)
    assert r2["started"] is False
    assert "已在运行" in r2["message"]
    w._thread.join(timeout=10.0)


def test_worker_passes_question_to_agent(monkeypatch):
    w = _ready_worker(monkeypatch)
    w.start(synthetic=True, question="只检测啸叫")
    w._thread.join(timeout=10.0)
    assert StubAgent.instances[0].question == "只检测啸叫"


# ---------- D18：ANC 实时残差路径 ----------

def _growing_sine(fs: int = 16000, seconds: float = 3.0, f0: float = 300.0) -> np.ndarray:
    """电平随时间线性增长的正弦：触发 detect_feedback 的啸叫判定。"""
    t = np.arange(int(fs * seconds)) / fs
    amp = np.linspace(0.01, 0.5, len(t))
    return (amp * np.sin(2.0 * np.pi * f0 * t)).astype(np.float64)


def test_tool_check_feedback_uses_anc_residual_when_running():
    """ANC 运行 + 残差可用 → 直接分析残差，不再重新采集。"""
    residual = _growing_sine()

    class FakeHandlers(ToolHandlers):
        def _capture_samples(self, duration_s, synthetic):
            raise AssertionError("ANC 运行且残差可用时不应重新采集")

    def provider():
        return {"state": "running", "phase": "cancelling", "f0": 120.0, "gain": 0.12}

    handlers = FakeHandlers(anc_status_provider=provider,
                            anc_residual_provider=lambda: residual)
    res = json.loads(handlers.call("check_feedback_or_howling", {}))
    assert res["source"] == "anc_residual"
    assert res["synthetic"] is False
    assert res["feedback"]["signal_class"] == "acoustic_feedback"
    assert res["anc_engine"]["running"] is True


def test_tool_check_feedback_falls_back_when_no_residual():
    """ANC 运行但无残差数据 → 回退实时采集。"""
    def provider():
        return {"state": "running", "phase": "cancelling"}

    handlers = ToolHandlers(anc_status_provider=provider,
                            anc_residual_provider=lambda: None)
    res = json.loads(handlers.call("check_feedback_or_howling",
                                   {"duration_s": 3.0, "synthetic": True}))
    assert res["source"] == "live_capture"
    assert res["synthetic"] is True
    # 合成打印机噪声不是啸叫；ANC 运行时可能被叠加 suspected_feedback 辅助判读
    assert res["feedback"]["is_howling"] is False
    assert res["feedback"]["signal_class"] in ("environment_noise", "suspected_feedback")


def test_tool_check_feedback_falls_back_when_residual_provider_errors():
    """残差 provider 抛异常 → 回退实时采集，不把异常透传给 Agent。"""
    def provider():
        return {"state": "running", "phase": "cancelling"}

    def bad_residual():
        raise RuntimeError("residual boom")

    handlers = ToolHandlers(anc_status_provider=provider,
                            anc_residual_provider=bad_residual)
    res = json.loads(handlers.call("check_feedback_or_howling",
                                   {"duration_s": 3.0, "synthetic": True}))
    assert res["source"] == "live_capture"
    assert res["synthetic"] is True
    assert res["feedback"]["is_howling"] is False


def test_tool_check_feedback_ignores_residual_when_anc_idle():
    """ANC 未运行 → 不碰残差，走实时采集。"""
    captured = {"n": 0}

    class FakeHandlers(ToolHandlers):
        def _capture_samples(self, duration_s, synthetic):
            captured["n"] += 1
            return _growing_sine(), False

    def residual():
        raise AssertionError("ANC 未运行时不应请求残差")

    handlers = FakeHandlers(anc_residual_provider=residual)
    res = json.loads(handlers.call("check_feedback_or_howling", {}))
    assert res["source"] == "live_capture"
    assert captured["n"] == 1


# ---------- D19：adjust_anc 工具描述与引擎边界一致 ----------

def test_adjust_anc_tool_description_matches_engine_bounds():
    """描述锁住 watchdog 增益边界：引擎改边界时若忘改工具描述，此测试会失败。"""
    from app.anc.live import LiveANCEngine
    from app.agent import AGENT_TOOLS

    tool = next(t for t in AGENT_TOOLS
                if t["function"]["name"] == "adjust_anc")
    desc = tool["function"]["description"]
    value_desc = tool["function"]["parameters"]["properties"]["value"]["description"]
    lo, hi = LiveANCEngine().watchdog_min_gain, LiveANCEngine().watchdog_max_gain
    assert f"{lo:g}" in desc and f"{hi:g}" in desc, f"描述缺少增益边界: {desc}"
    assert f"{lo:g}" in value_desc and f"{hi:g}" in value_desc, \
        f"value 参数描述缺少增益边界: {value_desc}"
