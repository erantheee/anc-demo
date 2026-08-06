# M1 测量协议：确认噪声大小与来源

## 目的

在 3D 打印机房间输出三样东西：

1. **噪声大小**：各测量点 A 加权 SPL（dB），形成房间噪声地图。
2. **噪声来源**：频谱峰值 + 谐波家族 → 归属到步进电机 / 风扇 / 结构共振等。
3. **是否需要降噪**：音调占比、主频段、安静区可行性 → 量化建议。

## 准备

1. 用 `scripts/calibrate-mic.py` 标定麦克风灵敏度（手机声级计 APP 或 94 dB 校准器）。
   - 方法：手机 APP 在麦克风处读 75 dB → 程序算出 offset，写入 `data/reports/calibration.json`。
2. 打印机固定位置，摄像头摆放好（M3 再做精确定位，本阶段记录人工坐标即可）。
3. 准备两个测量条件：**打印机静止**（环境底噪）与 **打印机打印中**。

## 步骤

1. 在房间里画网格（建议 0.5–1 m 间距，覆盖操作位与打印机周围）。
2. 每个网格点静止测 30 s（环境底噪），再打印中测 30 s。
3. 记录备注：打印机是否在打印、哪台设备在转、人的活动等。
4. 用 `scripts/measure.py` 自动完成录音 + 分析 + 生成报告。

```sh
.venv/bin/python scripts/measure.py \
  --grid "0,0 1,0 0,1 1,1" \
  --duration 30 \
  --out data/reports/room-001
```

## 报告格式

`data/reports/room-001/report.json`：

```json
{
  "grid": [{"x": 0, "y": 0, "spl_db_a": 62.1, "source_hits": ["fan_blade"]}],
  "noise_map": {"image": "noise-map.png", "max_db": 68.4, "min_db": 48.2},
  "dominant_sources": [
    {"source": "stepper", "confidence": 0.8, "freqs_hz": [120, 240, 360]}
  ],
  "recommendation": {
    "anc_worthwhile": true,
    "reasons": ["tonal_ratio_high", "dominant_freq_low_mid"],
    "suggested_quiet_zone_m": {"x": 1.2, "y": 1.5}
  }
}
```

## 判断规则（是否需要降噪）

| 条件 | 建议 |
|---|---|
| 主频 < 500 Hz 且音调占比 > 0.3 | ANC 值得做，预期 10–20 dB 音调降噪 |
| 主频 500 Hz–2 kHz 且音调占比高 | ANC 部分有效，建议 ANC + 被动吸音 |
| 主频 > 2 kHz 或宽带为主 | ANC 收益低，建议被动方案（隔音罩） |
| 安静区目标与打印机距离 > 3 m | 检查延迟与扩散衰减，一般降低预期 |

## 合成数据（无硬件自测）

```sh
.venv/bin/python scripts/measure.py --synthetic --out data/reports/synthetic-demo
```

用 `app/synth.py` 生成含步进音调 + 风扇宽带的合成录音，验证整条分析管线。
