"""CLI：用 Kimi 噪声自动检测 Agent 分析一段噪声（合成或真实麦克风采样）。

示例：
    .venv/bin/python scripts/agent_analyze.py --synthetic --duration 5
    .venv/bin/python scripts/agent_analyze.py --duration 3          # 真机麦克风采样
    MOONSHOT_API_KEY=sk-xxx .venv/bin/python scripts/agent_analyze.py --synthetic
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python scripts/agent_analyze.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import DEFAULT_QUESTION, NoiseDetectionAgent, ToolHandlers, get_api_key


def main() -> None:
    parser = argparse.ArgumentParser(description="Kimi 噪声自动检测 Agent（CLI）")
    parser.add_argument("--duration", type=float, default=3.0, help="采样时长（秒）")
    parser.add_argument("--synthetic", action="store_true", help="使用合成 3D 打印机噪声")
    parser.add_argument("--question", default=None, help="自定义检测指令（缺省用内置问题）")
    args = parser.parse_args()

    api_key = get_api_key()
    if not api_key:
        print("未配置 MOONSHOT_API_KEY（环境变量或项目根 .env），无法调用 Kimi。")
        raise SystemExit(2)

    print(f"[agent] 采样 {args.duration}s（synthetic={args.synthetic}）…")
    handlers = ToolHandlers()
    captured = handlers.capture_noise_sample(duration_s=args.duration, synthetic=args.synthetic)
    handlers.pre_captured = captured["analysis"]
    print(f"[agent] 采样完成，主频 {captured['analysis'].get('dominant_freq')} Hz，"
          f"SPL {captured['analysis'].get('spl_db') or captured['analysis'].get('rms_db')} dB")

    fb = json.loads(handlers.call("check_feedback_or_howling",
                                  {"synthetic": args.synthetic}))
    fd = fb.get("feedback", {})
    print(f"[agent] 啸叫检测: {fd.get('signal_class')}"
          f"（score={fd.get('howling_score')}，增长 {fd.get('growth_db_per_s')} dB/s，"
          f"候选频率 {fd.get('candidate_freq_hz')} Hz）")

    agent = NoiseDetectionAgent(api_key=api_key)
    t0 = time.time()
    outcome = agent.run(handlers, question=args.question or DEFAULT_QUESTION)
    elapsed = round(time.time() - t0, 1)

    print(f"[agent] {outcome['rounds']} 轮工具调用: {', '.join(outcome['tools_used']) or '无'}，"
          f"耗时 {elapsed}s")
    if outcome["result"] is None:
        print("[agent] 解析失败，原始回答：")
        print(outcome["raw_answer"])
        raise SystemExit(1)
    print(json.dumps(outcome["result"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
