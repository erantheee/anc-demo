"""FastAPI 端点集成测试（TestClient，不触发 lifespan / 不访问真实音频设备）。

覆盖本次新增的三个端点契约：
- /api/agent/status：配置态 + 状态透传
- /api/agent/analyze：未配置拒绝 / 已配置启动到 done / 并发拒绝
- /api/anc/live/control：引擎未运行报错 / 参数透传

用 monkeypatch 隔离 is_configured 与 ANC 引擎，全程无网络、无音频硬件。
"""
from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.agent import AgentStatus


@pytest.fixture
def client():
    # 不进入 lifespan：monitor.start() 需要音频设备，测试环境无硬件
    return TestClient(main.app)


@pytest.fixture(autouse=True)
def _clean_agent_worker():
    yield
    th = main.noise_agent._thread
    if th is not None and th.is_alive():
        th.join(timeout=5.0)
    with main.noise_agent._lock:
        main.noise_agent.status = AgentStatus(state="idle", configured=False)


# ---- /api/agent/status ----

def test_agent_status_unconfigured(client, monkeypatch):
    monkeypatch.setattr("app.agent.is_configured", lambda: False)
    main.noise_agent.status.configured = False
    r = client.get("/api/agent/status")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "idle"
    assert body["configured"] is False


# ---- /api/agent/analyze ----

def test_agent_analyze_rejected_when_unconfigured(client, monkeypatch):
    monkeypatch.setattr("app.agent.is_configured", lambda: False)
    monkeypatch.setattr("app.agent.get_api_key", lambda: None)
    r = client.post("/api/agent/analyze", json={"fresh_sample": True})
    assert r.status_code == 200
    body = r.json()
    assert body["started"] is False
    assert "MOONSHOT_API_KEY" in body["message"]
    assert client.get("/api/agent/status").json()["state"] == "idle"


def test_agent_analyze_runs_to_done(client, monkeypatch):
    from app.agent import DEFAULT_QUESTION

    runs = []

    class StubAgent:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key

        def run(self, handlers, question=None):
            runs.append(question)
            return {"rounds": 1, "tools_used": [], "raw_answer": "{}",
                    "result": {"summary": "ok"}}

    monkeypatch.setattr("app.agent.is_configured", lambda: True)
    monkeypatch.setattr("app.agent.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.agent.NoiseDetectionAgent", StubAgent)

    r = client.post("/api/agent/analyze", json={
        "fresh_sample": True, "synthetic": True, "duration_s": 0.5})
    assert r.status_code == 200
    assert r.json()["started"] is True

    th = main.noise_agent._thread
    assert th is not None
    th.join(timeout=10.0)
    body = client.get("/api/agent/status").json()
    assert body["state"] == "done"
    assert body["result"] == {"summary": "ok"}
    assert runs == [DEFAULT_QUESTION], "未传 question 时应使用默认问题"


def test_agent_analyze_rejects_concurrent_start(client, monkeypatch):
    class SlowAgent:
        def __init__(self, api_key, **kwargs):
            self.api_key = api_key

        def run(self, handlers, question=None):
            time.sleep(0.3)
            return {"rounds": 1, "tools_used": [], "raw_answer": "{}", "result": {}}

    monkeypatch.setattr("app.agent.is_configured", lambda: True)
    monkeypatch.setattr("app.agent.get_api_key", lambda: "test-key")
    monkeypatch.setattr("app.agent.NoiseDetectionAgent", SlowAgent)

    r1 = client.post("/api/agent/analyze", json={"synthetic": True, "duration_s": 0.3})
    assert r1.json()["started"] is True
    r2 = client.post("/api/agent/analyze", json={"synthetic": True})
    assert r2.json()["started"] is False
    assert "已在运行" in r2.json()["message"]


# ---- /api/anc/live/control ----

class FakeAncEngine:
    def __init__(self):
        self.calls = []

    def control(self, action, params):
        self.calls.append((action, params))
        return {"ok": True, "action": action}


def test_anc_live_control_errors_when_engine_stopped(client, monkeypatch):
    monkeypatch.setattr(main, "anc_live", None)
    r = client.post("/api/anc/live/control", json={"action": "set_gain", "value": 0.2})
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is False
    assert "未运行" in body["error"]


def test_anc_live_control_passthrough(client, monkeypatch):
    fake = FakeAncEngine()
    monkeypatch.setattr(main, "anc_live", fake)
    r = client.post("/api/anc/live/control", json={
        "action": "set_gain", "value": 0.2, "reason": "agent"})
    assert r.status_code == 200
    assert r.json()["ok"] is True
    assert fake.calls == [("set_gain", {"value": 0.2, "reason": "agent"})]


def test_anc_live_control_marshal_optional_params(client, monkeypatch):
    fake = FakeAncEngine()
    monkeypatch.setattr(main, "anc_live", fake)
    r = client.post("/api/anc/live/control", json={
        "action": "decrease_gain", "delta": 0.05})
    assert r.status_code == 200
    assert fake.calls == [("decrease_gain", {"delta": 0.05, "reason": ""})]


# ---- /api/anc/source：人声闸门 ----

def test_anc_source_flags_human_voice(client, monkeypatch):
    """人声：/api/anc/source 应标记 is_voice、不推荐 ANC、不预置 f0。"""
    from app.analyze import analyze
    from app.synth import speech_like

    x = speech_like(fs=48000, duration=5.0)
    report = analyze(x, 48000)
    monkeypatch.setattr(main.monitor, "last_report", lambda: report)
    monkeypatch.setattr(main.monitor, "last_samples", lambda: x)

    r = client.get("/api/anc/source")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["is_voice"] is True
    assert body["anc_worthwhile"] is False
    assert body["recommended_f0"] is None


def test_anc_source_ok_without_samples(client, monkeypatch):
    """无原始采样缓存时（如刚启动）：端点不崩，正常返回频谱建议。"""
    from app.synth import printer_noise

    x, _ = printer_noise(fs=16000, duration=4.0, seed=3)
    from app.analyze import analyze
    report = analyze(x, 16000)
    monkeypatch.setattr(main.monitor, "last_report", lambda: report)
    monkeypatch.setattr(main.monitor, "last_samples", lambda: None)

    r = client.get("/api/anc/source")
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True
    assert body["is_voice"] is False
