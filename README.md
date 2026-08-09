# 树莓派主动降噪 Demo（开放式 ANC 硬件）

完全运行在 Raspberry Pi 5 上的开放式主动降噪（Active Noise Cancellation）原型。
以 3D 打印机房间为第一个场景：先用摄像头对房间做空间建模、定位打印机，再测量确认
噪声大小与来源，判断是否需要降噪，最后用参考麦克风 + 误差麦克风 + 扬声器做 ANC，
量化降噪效果。架构上预留空调、硬件工程师风枪、风机、门外扫地机器人等迭代场景。

## 核心结论（先读这段）

- 主动降噪**只在一个局部点有效**：在"误差麦克风所在点"附近形成安静区，直径约 λ/10。
  100 Hz 约 30 cm，1 kHz 约 3 cm。**不能整房间降噪。**
- 3D 打印机噪声里，步进电机音调与风扇叶片频率是 ANC 甜点（预期降 10–20 dB）；
  风扇气动噪声等高频宽带成分 ANC 效果有限，主要靠被动方案。
- 因此本项目严格遵循"**先测量、后降噪**"：M1 先输出噪声地图、频谱、来源归属和
  "是否需要降噪"的量化建议，再进入 M2 的实时 ANC。

## 硬件

| 部件 | 型号示例 | 用途 | 状态 |
|---|---|---|---|
| 树莓派 | Pi 5 | 感知 + DSP + UI | 已有 |
| 麦克风（测量用） | USB 麦克风 | M1 网格录音 | 已有 |
| 摄像头 | USB 或 CSI | 空间建模 / ArUco 定位 | 已有 |
| 扬声器 | 有源音箱或功放板 | 反相声波输出 | 已有 |
| **I2S 编解码器 HAT** | WM8960（2 ADC + 2 DAC） | 实时 ANC 低延迟音频 | **需购买** |
| 参考/误差麦克风（ANC 用） | MEMS 麦接 WM8960 两路 ADC | 参考信号 + 误差信号 | 随 HAT 方案 |
| ArUco 标记 | A4 打印 | 打印机定位 | 打印即可 |

> 已有硬件型号确认后，请更新到 `docs/ENGINEERING_PLAN.md` 的硬件清单一节。

## 部署拓扑（已现场验证）

**最终形态：Pi 自播自听，Mac 只当浏览器/控制台。** 实时 ANC 的误差麦克风与扬声器
必须在同一台机器上（sounddevice 全双工流），因此 ANC 引擎跑在 **Pi** 上；Mac 的
8000 服务只提供浏览器仪表盘与 Mac 麦克风的可选监听，不参与降噪闭环。

| 设备 | 角色 | 服务 | 验证状态 |
|---|---|---|---|
| **Mac** | 浏览器 / 控制台（可选监听 Mac 麦克风） | `:8000` | 运行中，Mac 麦可采 |
| **Pi** | **监听 + ANC 执行**（Pi USB 麦 + Pi 音箱） | `:8001` | 运行中，`mic_ok: true`，实测 SPL -10.3 dB / 主频 152 Hz |

Pi 侧实测音频环境（2026-08-07）：

- USB 声卡：`JMTek USB PnP Audio Device`（`hw:2,0`），2ch in/out，采集正常；
- `~/.asoundrc` 的 `asym` 双工配置已生效（`anc_duplex` / `default` 均列出），
  真机 ANC 可直接开双工流；
- 8001 服务实时监控确认 `mic_ok: true`（USB 麦能采到真实噪声）；
- 该声卡实测环路延迟约 40ms，真机 ANC 用 `--mic-delay-ms 40`（Web 面板默认已是 40）。

> **当前阶段只做监听，不播 ANC**：本项目演示定位为"Pi 自播自听"——误差麦贴在
> "想要安静的点"，音箱放在旁边对着误差麦；安静区只在误差麦附近（λ/10），
> 安静区外会听到反相波与原噪声的干涉声，属正常现象。尚未启动实时降噪输出。
> 当前误差麦直接复用 USB 麦克风（M1 测量用那支），无需 WM8960 HAT 也能先跑
> M2 自参考谐波消除；WM8960 到位后再上低延迟两路（参考+误差）方案。

可选迭代元器件（按需购买）：

| 元器件 | 型号 | 作用 | 阶段 |
|---|---|---|---|
| 2D LiDAR | RPLIDAR A1 / LD19 | 房间 2D 地图，源/静音区坐标 | M3+ |
| 麦克风阵列 | ReSpeaker 4-Mic USB | 声源方向估计（DOA）辅助定位 | M3+ |
| ToF 距离传感器 | VL53L1X | 廉价测距补充 | M3+ |

## 运行

核心离线管线（无需硬件即可跑通 M1 分析，含合成 3D 打印机噪声）：

```sh
.venv/bin/python -m pip install -e ".[dev,audio,vision,plot,web]"
.venv/bin/python scripts/measure.py --synthetic --out data/reports/synthetic-demo
```

真实测量：

```sh
.venv/bin/python scripts/calibrate-mic.py --known-spl 75   # 用手机声级计 APP 对照
.venv/bin/python scripts/measure.py --grid "0,0 1.5,0 0,1.5 1.5,1.5" --duration 30 --out data/reports/room-001
```

> **麦克风标定（Web 也可做）**：默认仪表盘显示的是 dBFS 相对值（数值偏小、量纲不对）。
> 用手机声级计 APP 在麦克风处读出真实 SPL，再在 Web 面板"实时噪声 → 麦克风标定"里
> 填入读数点"开始标定"，或 CLI 跑
> `.venv/bin/python scripts/calibrate_mic.py --known-spl 75`。系统把偏移写入
> `data/reports/calibration.json` 并立即生效，之后仪表盘 / 网格测量 / ANC 报告都显示
> 绝对 dB SPL。注意 Mac 内置麦克风有 AGC，建议用 USB 麦克风做测量。

实时 ANC（M2，自参考谐波消除，无需参考麦克风 / I2S 编解码器）：

```sh
.venv/bin/python scripts/run_anc_live.py --list                 # 列出输入/输出设备
.venv/bin/python scripts/run_anc_live.py --synthetic --duration 15   # 无硬件自测
.venv/bin/python scripts/run_anc_live.py --in-device "USB Mic" --out-device "3.5mm" \
    --fs 48000 --baseline 5 --duration 60                        # 现场真机
```

> 扬声器→误差麦延迟补偿：`--mic-delay-ms`（默认 5.0ms）让反相输出"向前预测"，
> 对准扬声器声波实际到达误差麦时的噪声相位。麦克风紧贴扬声器约 0.5–1ms，相距
> 0.5m 约 2ms，1.5m 约 5ms；现场可先 1ms 起逐步调，观察误差麦实时 SPL 最低点。
>
> **USB 声卡必须先实测环路延迟**：USB 缓冲延迟（通常 10–60ms）远大于声学延迟，
> 相位错开会让消除失效。用 `scripts/measure_loop_delay.py` 实测后把结果填进
> `--mic-delay-ms`（Web 仪表盘"麦克风延迟"）。本机（USB PnP Audio Device）实测
> 约 40ms，Web 默认已改为 40。
>
> **延迟自动扫描（默认开启）**：cancelling 开始时引擎会在 20/30/40/50/60ms
> 间试扫约 2 秒，选误差麦 RMS 最低的偏移（状态栏显示"自动校准 X ms"）。
> 因为 `sd.Stream` 的缓冲布局与 `playrec` 实测值可能不同，自动校准比手工
> 填固定值更可靠。可用 `--no-auto-scan` 关闭。
>
> **反馈中和（默认开启，`--feedback-cancel-gain 0.3`）**：从误差麦信号中减去
> "扬声器回声估计"，避免自适应滤波把扬声器自己的反相输出当噪声去追（正反馈
> 自激会让误差麦总声压净增）。耦合增益因设备/摆放而异，过大过小都会劣化效果。
>
> **A/B 报告以误差麦真实信号为准**：`cancelling_spl_db` 是 ANC 开启期间误差麦
> 实际采到的声音（含反相波回采），不是数字域理想误差。报告中的**音调降噪
> （tone_reduction_db，顶部谱峰中位降噪）**是谐波消除的主要指标；宽带数会被
> 房间本底噪声与扬声器回采干扰，噪声非稳态时基线/降噪段对比会上下浮动 ±2–6dB。
> 若音调峰降噪 10–20dB 但宽带数为正，通常说明噪声源以宽带为主——谐波消除
> 只对周期分量有效，此时请用手机/电脑播放**稳定纯音**（如 150–300Hz）贴近
> 误差麦再测，总声压即会随音调一并下降。

> **单个 USB 音频设备的全双工适配**：若只有一支"喇叭+麦克风"二合一 USB 设备，
> PortAudio 无法在该设备上直接开全双工流（同时录放报 Invalid number of channels）。
> 在树莓派上写入 `~/.asoundrc` 用 ALSA `asym` 插件把录/放拆成两个独立 PCM，
> 即可让现有引擎开双工流（下述配置实测有效，且 `--mic-delay-ms` 用 40 附近值）：
>
> ```
> pcm.!default {
>     type asym
>     playback.pcm { type plug; slave.pcm "hw:2,0" }   # 2 换成实际声卡号（aplay -l 查看）
>     capture.pcm  { type plug; slave.pcm "hw:2,0" }
> }
> ```
>
> 注意此配置也会改变系统默认录音/播放指向（aplay/arecord 不带 -D 时也走它）。
> 若麦克风采集削波（录音峰值接近满量程），用 `amixer -c <card> set Mic cap <值>`
> 调低采集增益。

> **无麦克风保护**：任何测量/ANC 前都会校验输入设备与信号有效性。若未接入
> 麦克风（或线缆悬空），系统**不会**报告误导性分贝，而是明确报错
> "未检测到任何麦克风/输入设备"（Web 仪表盘实时噪声区显示"无麦克风"，SPL 显示 `--`）。
> 这是为避免把无输入/悬空输入的电平误当成环境噪声。
>
> **防啸叫 / 输出安全（办公室演示）**：反相输出经过三道硬保护，任何情况下都不会
> 把扬声器打得很响：
> 1. **绝对硬上限**：峰值限幅 `±clip_level`（默认 0.6，≈ -4.4 dBFS，扬声器中等音量档），
>    与输入无关——即使 NLMS 权重发散 / 声学正反馈，输出峰值也不可能超过它；
> 2. **RMS 比例门控**：输出 RMS ≤ ratio × 输入 RMS（默认 ratio=2.5，>1 才能产生真实降噪），
>    "输入小声 → 反相输出同步小声"，不做固定大音量去盖；
> 3. **安静静音**：输入低于 -60 dBFS 时输出全零。
>
> **规则式自动调节 watchdog（默认开启）**：cancelling 阶段引擎每 ~0.5s 评估一次误差麦
> SPL 趋势，两条规则自动调增益：
> - **啸叫 / 正反馈**：SPL 持续快速上升（> 3 dB/s）→ 立即降增益（Web 面板"自动调节日志"）；
> - **降噪不足**：进入降噪 3s 后 SPL 仍高于基线 1.5 dB → 逐步增增益（上限 1.0）。
> 这是不依赖 LLM 的亚秒级兜底，先于 Agent 保证现场安全。Web/CLI 可用
> `--no-watchdog`（或取消勾选"自动调节"）关闭。
>
> **Agent 可写闭环控制**：ANC 运行中，Kimi 噪声检测 Agent 除了检测啸叫/降噪效果，
> 还可以通过 `adjust_anc` 工具直接调节引擎（降增益/增增益/调延迟/停止），每次调节
> 记录在检测结论的 `anc_adjustments` 与引擎的"自动调节日志"里。规则 watchdog 兜底
> 保安全，Agent 做慢速全局调优，两者共同构成闭环。
>
> 默认 `gain=0.5`、权重范数上限 2.0（正反馈发散时自动缩回）。想更保守可把 Web 面板
> "输出增益"调到 0.2–0.35。演示时请把麦克风与扬声器拉开 0.5m 以上、扬声器不要正对
> 麦克风；测试噪声用稳态周期声（纯音 / 风扇 / 步进电机声）效果最好，音乐/人声这类
> 时变宽带信号谐波消除收益有限。

Web UI / API（沿用了此前 Web 服务习惯）：

```sh
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

打开 `http://<设备IP>:8000` 即仪表盘，包含：

- **实时噪声**：SPL 大数字、主频、来源猜测、低/中/高频段、频谱图（每 3 秒采样一次）
- **Pi 状态**：工作中/测量中/异常徽标、CPU 温度、运行时长、采样时间
- **房间建模**：配置网格（范围/步长/每点时长）→ 后台测量 → 交互式噪声地图，
  自动建议静音区（圆 = 安静区直径），点选地图任意点查看 ANC 可行性
  （距离 / 传播延迟 / 安静区直径 / 结论）
- **ANC 实时降噪**：合成模式或真机模式；采基线（ANC off）→ 自动估计基频 →
  实时输出反相谐波；实时 SPL 曲线与总降噪量，完成后给出 A/B 报告
  （宽带 / A 加权 / 各音调峰值降低 dB）

合成模式（无硬件自测）：仪表盘勾选"合成模式"，或用环境变量

```sh
ANC_SYNTHETIC=1 .venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Kimi 噪声自动检测 Agent（AI 噪声检测）

用 Kimi K3 驱动的"噪声自动检测 Agent"：读取频谱分析 → 识别噪声源 → 判断是否值得
降噪 → 给出参考麦克风与静音区建议。在规则式匹配（`app/source_id.py`）之上叠加
LLM 判读，输出结构化结论（JSON）。

**环境噪声 vs 啸叫 vs ANC**：Agent 通过 `check_feedback_or_howling` 工具检测当前
信号是**环境噪声**还是**声反馈/啸叫**（麦克风→扬声器→麦克风闭环自激振荡，频谱上
是电平随时间增长的窄带强音调），并结合 `get_anc_status` 确认 ANC 引擎是否在运行：
- 环境噪声 → 建议是否值得 ANC（低频音调是甜点）；
- 啸叫 → 判定 `is_howling=true`，`anc_worthwhile=false`，建议停止 ANC / 降低输出
  增益 / 拉开扬声器与麦克风距离（啸叫是系统自身的问题，不是要消除的目标）。

**安装依赖与配置 Key**（`openai` 为可选依赖，需单独安装）：

```sh
.venv/bin/python -m pip install -e ".[agent]"
export MOONSHOT_API_KEY="你的_KIMI_API_KEY"      # 或写入项目根 .env：MOONSHOT_API_KEY=sk-xxx
```

> API Key 从环境变量 `MOONSHOT_API_KEY`（或 `KIMI_API_KEY`）读取；也可在项目根放
> `.env` 文件（已 gitignore，不会提交）。未配置 Key 时 Agent 不可用，Web 面板会
> 显示"未配置"，其余功能不受影响。模型可用 `KIMI_AGENT_MODEL` 覆盖（默认
> `kimi-k3`），推理强度用 `KIMI_AGENT_REASONING`（`low`/`high`/`max`，默认 `low`）。

**Web 仪表盘**：启动 Web 服务后在"AI 噪声检测"卡片点"开始 AI 检测"。可勾选
"先重新采样"（采集一段最新噪声再分析）与"合成模式"（无麦克风时用合成打印机噪声）。
结论卡片展示：噪声源 / 置信度 / 主频与建议基频 / 是否值得降噪 / 判据 / 建议动作 /
参考麦克风与静音区建议。Agent 在后台线程运行，期间自动暂停监控采样。

**CLI 用法**：

```sh
.venv/bin/python scripts/agent_analyze.py --synthetic --duration 5   # 合成噪声
.venv/bin/python scripts/agent_analyze.py --duration 3               # 真实麦克风
```

**外部 Agent / 脚本的自动增益接口**（用户提供的检测 Agent 可周期调用）：

```sh
# 设定 ANC 目标输出增益（运行时生效，不打断正在跑的 ANC）
curl -X POST http://<设备IP>:8000/api/anc/live/gain \
  -H "Content-Type: application/json" -d '{"gain": 0.08}'
# 返回 {"ok": true, "state": ..., "gain": 0.08, ...}（含完整 status()）
```

- 契约：`POST /api/anc/live/gain`，body `{"gain": <float>}`。`gain` 会被钳制到
  [0.02, 1.0]，超出自动截断；引擎另有硬性兜底（±clip_level 硬限幅 0.6 + RMS
  比例门控 ratio 2.5），外部 Agent 无法把输出推到危险水平。
- 更细粒度控制（相对增减 / 调延迟 / 停止）走 `POST /api/anc/live/control`，
  `action` ∈ `set_gain | increase_gain | decrease_gain | set_mic_delay_ms | stop`。
- 查询当前 ANC 状态（含生效 gain 与 watchdog 日志）：`GET /api/anc/live/status`。
- 内置规则 watchdog 默认开启（不调用外部 Agent 时也会自动跟随输入电平调增益）；
  外部 Agent 是可选的增强层，不是必需。

## 分阶段计划

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 硬件准备与校准：音频 I/O 验证、麦克风灵敏度标定 | 待启动 |
| M1 | 噪声测量与来源确认：网格录音 → 噪声地图 + 频谱 + 来源归属 + 是否需要降噪建议 | **第一步** |
| M1.5 | Web 仪表盘：实时噪声 + Pi 状态 + 交互式噪声地图与静音区建议 | **已完成** |
| M2 | ANC Demo：实时谐波消除（自参考，单误差麦 + 扬声器），A/B 评估；FXLMS 离线模拟预留 | **已完成** |
| M3 | 空间建模与位置闭环：摄像头建模房间 → 定位打印机 → 引导静音区与 ANC 参数 | 待启动 |
| M4 | 迭代场景：空调 / 风枪 / 风机 / 门外扫地机器人 Profile 抽象 | 待启动 |

## 文档

- `docs/ENGINEERING_PLAN.md` — 工程计划与硬件清单
- `docs/ANC_THEORY.md` — ANC 原理、安静区、延迟预算、FXLMS
- `docs/MEASUREMENT_PROTOCOL.md` — M1 测量流程与报告格式
- `TODOS.md` — 未进入当前 Demo 的功能
