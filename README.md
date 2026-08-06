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
```

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
