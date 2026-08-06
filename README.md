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
.venv/bin/python scripts/run-anc.py --mode harmonic        # 实时 ANC（需要 I2S 编解码器）
```

Web UI / API（沿用吉他 demo 习惯）：

```sh
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## 分阶段计划

| 里程碑 | 内容 | 状态 |
|---|---|---|
| M0 | 硬件准备与校准：音频 I/O 验证、麦克风灵敏度标定 | 待启动 |
| M1 | 噪声测量与来源确认：网格录音 → 噪声地图 + 频谱 + 来源归属 + 是否需要降噪建议 | **第一步** |
| M2 | ANC Demo：参考麦 + 误差麦 + 扬声器，FXLMS / 谐波消除，A/B 评估 | 待启动 |
| M3 | 空间建模与位置闭环：摄像头建模房间 → 定位打印机 → 引导静音区与 ANC 参数 | 待启动 |
| M4 | 迭代场景：空调 / 风枪 / 风机 / 门外扫地机器人 Profile 抽象 | 待启动 |

## 文档

- `docs/ENGINEERING_PLAN.md` — 工程计划与硬件清单
- `docs/ANC_THEORY.md` — ANC 原理、安静区、延迟预算、FXLMS
- `docs/MEASUREMENT_PROTOCOL.md` — M1 测量流程与报告格式
- `TODOS.md` — 未进入当前 Demo 的功能
