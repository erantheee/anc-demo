import {
  Callout,
  Card,
  CardBody,
  CardHeader,
  Grid,
  H1,
  H2,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
} from "cursor/canvas";

function FlowCard({ title, detail }: { title: string; detail: string }) {
  return (
    <Card>
      <CardHeader>{title}</CardHeader>
      <CardBody>
        <Text size="small" tone="secondary">
          {detail}
        </Text>
      </CardBody>
    </Card>
  );
}

function Arrow() {
  return <Text tone="tertiary">→</Text>;
}

export default function PiAncArchitecture() {
  return (
    <Stack gap={16}>
      <H1>树莓派主动降噪 Demo — 架构总览</H1>
      <Text tone="secondary">
        独立新项目：先测量确认 3D 打印机噪声的大小与来源，再在其所在房间实现主动降噪（ANC），并预留空调、风枪、风机、扫地机器人等迭代场景。
      </Text>

      <Callout tone="info" title="核心物理约束：安静区与噪声类型">
        主动降噪只在误差麦克风所在点附近形成安静区（直径约 λ/10），无法整房间降噪。3D 打印机噪声中，步进电机音调成分与风扇叶片频率是 ANC 甜点；高频宽带成分（气动噪声）主要靠被动方案。因此"先测量确认来源"是正确路径。
      </Callout>

      <Grid columns={3} gap={16}>
        <Stat value="~30 cm" label="安静区直径 @ 100 Hz" />
        <Stat value="~3 cm" label="安静区直径 @ 1 kHz" />
        <Stat value="10–20 dB" label="目标降噪（音调成分）" />
      </Grid>

      <H2>系统架构</H2>

      <Card>
        <CardHeader trailing={<Pill size="sm" active>感知</Pill>}>
          摄像头 → 空间建模 → 打印机定位
        </CardHeader>
        <CardBody>
          <Row gap={12} align="center" wrap>
            <FlowCard title="CSI / USB 摄像头" detail="房间俯视或斜视" />
            <Arrow />
            <FlowCard title="ArUco 标记 + OpenCV" detail="打印机位置 x,y,z + 朝向" />
            <Arrow />
            <FlowCard title="空间建模" detail="房间坐标系 + 平面检测" />
            <Arrow />
            <FlowCard title="静音区选择" detail="用户点选 / 系统推荐" />
            <Arrow />
            <FlowCard title="打印状态检测" detail="打印中自动开启 ANC" />
          </Row>
          <Row gap={12} align="center">
            <Text size="small" tone="secondary">
              迭代：LiDAR（RPLIDAR）2D 房间地图、麦克风阵列 DOA 辅助定位
            </Text>
          </Row>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill size="sm" active>ANC 主环</Pill>}>
          参考麦克风 → FXLMS → 扬声器
        </CardHeader>
        <CardBody>
          <Row gap={12} align="center" wrap>
            <FlowCard title="参考麦克风" detail="靠近打印机，捕获噪声" />
            <Arrow />
            <FlowCard title="WM8960 I2S 编解码器" detail="2 ADC + 1 DAC，ALSA 低延迟" />
            <Arrow />
            <FlowCard title="FXLMS / 谐波消除" detail="树莓派直跑，3–10 ms 延迟" />
            <Arrow />
            <FlowCard title="扬声器" detail="反相声波（有源音箱）" />
          </Row>
          <Row gap={12} align="center">
            <Text size="small" tone="secondary">
              误差麦克风（安静点）→ 残差反馈 → 自适应滤波器收敛
            </Text>
          </Row>
        </CardBody>
      </Card>

      <Card>
        <CardHeader trailing={<Pill size="sm" active>测量与评估</Pill>}>
          噪声地图 → 来源归属 → ANC 前后对比
        </CardHeader>
        <CardBody>
          <Row gap={12} align="center" wrap>
            <FlowCard title="网格测量录音" detail="房间多点，多个打印阶段" />
            <Arrow />
            <FlowCard title="SPL / 频谱 / 噪声图" detail="确认噪声大小与来源" />
            <Arrow />
            <FlowCard title="ANC 可行性判断" detail="音调占比 → 是否需要降噪" />
            <Arrow />
            <FlowCard title="A/B 评估" detail="ANC 开关对比，dB 差" />
          </Row>
        </CardBody>
      </Card>

      <H2>软件模块</H2>
      <Table
        headers={["模块", "职责"]}
        rows={[
          ["capture.py", "音频采集（ALSA / USB / I2S），统一录音接口"],
          ["analyze.py", "SPL、FFT、频谱峰值、音调占比"],
          ["source_id.py", "噪声源归属（步进 / 风扇 / 共振）"],
          ["noise_map.py", "房间网格测量 → 噪声地图"],
          ["position.py", "摄像头 → ArUco 打印机定位"],
          ["anc/fxlms.py", "Filtered-x LMS 实时降噪"],
          ["anc/harmonic.py", "周期噪声谐波消除"],
          ["evaluate.py", "ANC 前后 A/B、dB 差、A 加权"],
          ["main.py", "Web UI / API（:8000，沿用吉他 demo 习惯）"],
        ]}
      />

      <H2>里程碑</H2>
      <Table
        headers={["里程碑", "内容", "状态"]}
        rows={[
          [
            "M0",
            "硬件准备与校准：音频 I/O 验证、麦克风灵敏度标定",
            <Pill>待启动</Pill>,
          ],
          [
            "M1",
            "噪声测量与来源确认：网格录音 → 噪声地图 + 频谱 + 来源归属 + 是否需要降噪建议",
            <Pill active>第一步</Pill>,
          ],
          [
            "M2",
            "ANC Demo：参考麦 + 误差麦 + 扬声器，FXLMS 音调消除，A/B 评估",
            <Pill>待启动</Pill>,
          ],
          [
            "M3",
            "位置感知闭环：摄像头定位打印机 → 自动引导静音区与 ANC 参数",
            <Pill>待启动</Pill>,
          ],
          [
            "M4",
            "迭代场景：空调 / 风枪 / 风机 / 门外扫地机器人 Profile 抽象",
            <Pill>待启动</Pill>,
          ],
        ]}
      />

      <H2>迭代场景</H2>
      <Table
        headers={["场景", "噪声类型", "ANC 可行性", "关键设计点"]}
        rows={[
          ["3D 打印机（v1）", "步进音调 + 风扇宽带", "音调高 / 宽带中", "参考麦近打印机；静音区在操作位"],
          ["空调", "压缩机 50/60 Hz 谐波", "高", "低频安静区大；参考麦在出风口"],
          ["硬件工程师风枪", "宽带强噪声、间歇", "低–中", "高 SPL；防热距离；参考麦近喷嘴"],
          ["风机", "叶片频率 + 宽带", "中", "与打印机类似；参考麦在进风口"],
          ["门外扫地机器人", "电机 + 结构传导", "中", "声源移动；参考麦在门 / 墙边"],
        ]}
      />
    </Stack>
  );
}
