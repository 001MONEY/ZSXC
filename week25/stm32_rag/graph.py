"""Neo4j 器件知识图谱：把 `mcu.py` 的静态事实（器件 / 引脚 / 定时器通道 / USART / DMA / 总线 / 中断）
建成一张属性图，补上向量检索不擅长的**多跳关系查询**。

为什么不直接让模型背表：

- `PA8 → TIM1_CH1 → DMA1_Channel2 → DMA1_Channel2_IRQn` 这种链路，一次 Cypher 就查全，
  模型不用连着做三四次猜；
- 引脚复用有没有、DMA 通道是几号、挂在 APB1 还是 APB2，都能在**写代码之前**问清楚；
- 图与 `mcu.py` 可以互相校验（`verify()`），表改了忘了改图会被立刻发现。

设计取舍：图里的数据**只从 `mcu.py` 生成**（单一数据源），不手工维护第二份事实。
从 PDF 抽实体（chunk ↔ 符号 ↔ 章节）属于 GraphRAG 的下一层，见 README 第 8.4 节。

建图与查询：

    python -m stm32_rag graph --build            # 建图（可重复执行，幂等）
    python -m stm32_rag graph --pin PA8
    python -m stm32_rag graph --usart USART1
    python -m stm32_rag graph --timer TIM3
    python -m stm32_rag graph --pwm-pins
    python -m stm32_rag graph --device STM32F103C8
    python -m stm32_rag graph --stats
    python -m stm32_rag graph --verify
    python -m stm32_rag graph --cypher "MATCH (p:Pin)-[:TIM_CH]->(t:Timer) RETURN p.name, t.name LIMIT 5"
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .mcu import (
    APB2_TIMERS,
    DEVICES,
    DMA_MAP,
    TIM_CHANNELS,
    TIMER_IRQ,
    TIMERS_BY_DEVICE,
    USART_IRQ,
    USART_ON_APB2,
    USART_PINS,
    Pin,
    device_pins,
    device_usarts,
    resolve_device,
)

# 图里用到的标签，按这个顺序建约束 / 清理
LABELS = ("Device", "Pin", "Timer", "Usart", "DmaChannel", "Bus", "Irq")

# 只读查询的黑名单：给 Agent 的 `--cypher` 用，避免模型把图写坏
_WRITE_WORDS = re.compile(r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|CALL\s+apoc\.(?!meta|coll|text|date|number|map|categor))", re.I)


@dataclass
class GraphResult:
    """一次建图的结果摘要。"""

    nodes: dict[str, int]
    relationships: dict[str, int]
    skipped: list[str]

    def summary(self) -> str:
        node_text = "、".join(f"{label} {count}" for label, count in self.nodes.items() if count)
        rel_text = "、".join(f"{name} {count}" for name, count in self.relationships.items() if count)
        lines = [f"节点：{node_text or '无'}", f"关系：{rel_text or '无'}"]
        if self.skipped:
            lines.append("跳过（图谱未建模的外设）：" + "、".join(self.skipped))
        return "\n".join(lines)


class GraphStore:
    """Neo4j 连接 + 建图 + 结构化查询。"""

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        database: str | None = None,
        timeout: float | None = None,
    ) -> None:
        try:
            from neo4j import GraphDatabase  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover - 依赖缺失
            raise RuntimeError(
                "缺少 neo4j 驱动：uv pip install --python <venv>/Scripts/python.exe neo4j"
            ) from exc
        self.uri = uri or NEO4J_URI
        self.database = database or NEO4J_DATABASE
        self._driver = GraphDatabase.driver(
            self.uri,
            auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
            **(({"connection_timeout": timeout}) if timeout else {}),
        )

    # ------------------------------------------------------------ 连接
    def ping(self) -> bool:
        """服务在不在（短超时，避免 CLI / Agent 卡住）。"""
        try:
            self._driver.verify_connectivity()
            return True
        except Exception:
            return False

    def close(self) -> None:
        self._driver.close()

    def __enter__(self) -> GraphStore:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def run(self, cypher: str, params: dict | None = None, limit: int = 100) -> list[dict]:
        """执行只读查询（写操作会被拒绝）。"""
        if _WRITE_WORDS.search(cypher):
            raise ValueError("只允许只读查询：CREATE/MERGE/DELETE/SET 等写操作已被拒绝")
        with self._driver.session(database=self.database) as session:
            result = session.run(cypher, params or {})
            rows = []
            for i, record in enumerate(result):
                if i >= limit:
                    break
                rows.append(dict(record))
            return rows

    # ------------------------------------------------------------ 建图
    def build(self, reset: bool = False) -> GraphResult:
        """把 `mcu.py` 的事实写进图；`MERGE` 保证幂等。"""
        if reset:
            self.reset()
        with self._driver.session(database=self.database) as session:
            for label in LABELS:
                session.run(
                    f"CREATE CONSTRAINT {label.lower()}_name IF NOT EXISTS "
                    f"FOR (n:{label}) REQUIRE n.name IS UNIQUE"
                )

            # 关系全部重建：只 MERGE 不清理的话，表改小（如 C8 去掉 UART4）会留下旧边
            session.run(
                """
                MATCH ()-[r:HAS_PIN|HAS_TIMER|HAS_USART|TIM_CH|USART_ROLE|DMA|CLOCKED_BY|IRQ]->()
                DELETE r
                """
            )

            skipped: list[str] = []

            # ---- 节点：器件 ----
            devices = [
                {
                    "name": d.name,
                    "define": d.define,
                    "flash_kb": d.flash_kb,
                    "ram_kb": d.ram_kb,
                    "ports": list(d.ports),
                    "startup": d.startup,
                }
                for d in DEVICES.values()
            ]
            session.run(
                """
                UNWIND $rows AS row
                MERGE (d:Device {name: row.name})
                SET d.define = row.define, d.flash_kb = row.flash_kb,
                    d.ram_kb = row.ram_kb, d.ports = row.ports, d.startup = row.startup
                """,
                rows=devices,
            )

            # ---- 节点：引脚（按封装真实引出的脚，不是端口×16）----
            pins = sorted({pin.name for d in DEVICES.values() for pin in device_pins(d)})
            session.run(
                """
                UNWIND $rows AS row
                MERGE (p:Pin {name: row.name})
                SET p.port = row.port, p.number = row.number
                """,
                rows=[{"name": name, "port": name[1], "number": int(name[2:])} for name in pins],
            )

            # ---- 节点：定时器 / USART / DMA / 总线 / 中断 ----
            timers = sorted({t for group in TIMERS_BY_DEVICE.values() for t in group} | set(TIM_CHANNELS))
            session.run(
                """
                UNWIND $rows AS row
                MERGE (t:Timer {name: row.name})
                SET t.advanced = row.advanced, t.bus = row.bus
                """,
                rows=[{"name": t, "advanced": t in ("TIM1", "TIM8"), "bus": "APB2" if t in APB2_TIMERS else "APB1"} for t in timers],
            )
            session.run(
                "UNWIND $rows AS row MERGE (u:Usart {name: row.name}) SET u.bus = row.bus",
                rows=[{"name": u, "bus": "APB2" if u in USART_ON_APB2 else "APB1"} for u in USART_PINS],
            )
            dma_channels = sorted(set(DMA_MAP.values()))
            session.run(
                "UNWIND $rows AS row MERGE (c:DmaChannel {name: row.name}) SET c.irq = row.irq",
                rows=[{"name": c, "irq": f"{c}_IRQn"} for c in dma_channels],
            )
            session.run("UNWIND $rows AS row MERGE (b:Bus {name: row.name})", rows=[{"name": "APB1"}, {"name": "APB2"}])
            irqs = sorted(set(TIMER_IRQ.values()) | set(USART_IRQ.values()))
            session.run("UNWIND $rows AS row MERGE (i:Irq {name: row.name})", rows=[{"name": i} for i in irqs])

            # ---- 关系：器件拥有引脚 / 定时器 / 串口 ----
            for device in DEVICES.values():
                session.run(
                    """
                    MATCH (d:Device {name: $device})
                    UNWIND $pins AS pin_name
                    MATCH (p:Pin {name: pin_name})
                    MERGE (d)-[:HAS_PIN]->(p)
                    """,
                    device=device.name,
                    pins=[pin.name for pin in device_pins(device)],
                )
                session.run(
                    """
                    MATCH (d:Device {name: $device})
                    MATCH (t:Timer) WHERE t.name IN $timers
                    MERGE (d)-[:HAS_TIMER]->(t)
                    """,
                    device=device.name,
                    timers=list(TIMERS_BY_DEVICE.get(device.name, ())),
                )
                # 串口的 TX/RX 引脚都在封装上，才算这颗片子有
                session.run(
                    """
                    MATCH (d:Device {name: $device})
                    MATCH (u:Usart) WHERE u.name IN $usarts
                    MERGE (d)-[:HAS_USART]->(u)
                    """,
                    device=device.name,
                    usarts=device_usarts(device),
                )

            # ---- 关系：引脚 ↔ 定时器通道 ----
            for timer, channels in TIM_CHANNELS.items():
                for channel, channel_pins in channels.items():
                    session.run(
                        """
                        UNWIND $pins AS pin_name
                        MATCH (p:Pin {name: pin_name}) MATCH (t:Timer {name: $timer})
                        MERGE (p)-[r:TIM_CH {channel: $channel}]->(t)
                        """,
                        pins=list(channel_pins),
                        timer=timer,
                        channel=channel,
                    )

            # ---- 关系：引脚 ↔ USART 角色 ----
            for usart, roles in USART_PINS.items():
                for role, role_pins in roles.items():
                    session.run(
                        """
                        UNWIND $pins AS pin_name
                        MATCH (p:Pin {name: pin_name}) MATCH (u:Usart {name: $usart})
                        MERGE (p)-[r:USART_ROLE {role: $role}]->(u)
                        """,
                        pins=[pin.name for pin in role_pins],
                        usart=usart,
                        role=role,
                    )

            # ---- 关系：DMA 硬连线（只建模定时器 / USART）----
            for (peripheral, role), channel in DMA_MAP.items():
                if peripheral.startswith("TIM"):
                    session.run(
                        """
                        MATCH (t:Timer {name: $name}) MATCH (c:DmaChannel {name: $channel})
                        MERGE (t)-[r:DMA {channel: $role}]->(c)
                        """,
                        name=peripheral,
                        role=role,
                        channel=channel,
                    )
                elif peripheral.startswith("USART") or peripheral.startswith("UART"):
                    session.run(
                        """
                        MATCH (u:Usart {name: $name}) MATCH (c:DmaChannel {name: $channel})
                        MERGE (u)-[r:DMA {role: $role}]->(c)
                        """,
                        name=peripheral,
                        role=role,
                        channel=channel,
                    )
                else:
                    skipped.append(f"{peripheral}.{role}")

            # ---- 关系：时钟总线 / 中断 ----
            session.run(
                """
                MATCH (t:Timer) MATCH (b:Bus {name: t.bus})
                MERGE (t)-[:CLOCKED_BY]->(b)
                """
            )
            session.run(
                """
                MATCH (u:Usart) MATCH (b:Bus {name: u.bus})
                MERGE (u)-[:CLOCKED_BY]->(b)
                """
            )
            for timer, irq in TIMER_IRQ.items():
                session.run(
                    "MATCH (t:Timer {name: $timer}) MATCH (i:Irq {name: $irq}) MERGE (t)-[:IRQ]->(i)",
                    timer=timer,
                    irq=irq,
                )
            for usart, irq in USART_IRQ.items():
                session.run(
                    "MATCH (u:Usart {name: $usart}) MATCH (i:Irq {name: $irq}) MERGE (u)-[:IRQ]->(i)",
                    usart=usart,
                    irq=irq,
                )

            counts = self._counts(session)
        return GraphResult(nodes=counts["nodes"], relationships=counts["relationships"], skipped=skipped)

    def reset(self) -> None:
        """只删本模块建的那几类节点（不会清库里的其它数据）。"""
        with self._driver.session(database=self.database) as session:
            for label in LABELS:
                session.run(f"MATCH (n:{label}) DETACH DELETE n")

    def stats(self) -> dict:
        with self._driver.session(database=self.database) as session:
            return self._counts(session)

    @staticmethod
    def _counts(session) -> dict:
        nodes: dict[str, int] = {}
        for label in LABELS:
            nodes[label] = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()["c"]
        rels: dict[str, int] = {}
        rows = session.run(
            "MATCH ()-[r]->() RETURN type(r) AS name, count(r) AS c ORDER BY name"
        )
        for row in rows:
            rels[row["name"]] = row["c"]
        return {"nodes": nodes, "relationships": rels}

    # ------------------------------------------------------------ 查询
    def pin_facts(self, pin: str, device: str | None = None) -> dict | None:
        """一个引脚能干什么：定时器通道、串口角色；都没有就是普通 GPIO。"""
        rows = self.run(
            """
            MATCH (p:Pin {name: $pin})
            OPTIONAL MATCH (p)-[tc:TIM_CH]->(t:Timer)
            OPTIONAL MATCH (p)-[ur:USART_ROLE]->(u:Usart)
            RETURN p.name AS pin, p.port AS port, p.number AS number,
                   [x IN collect(DISTINCT {timer: t.name, channel: tc.channel, advanced: t.advanced})
                      WHERE x.timer IS NOT NULL] AS timers,
                   [x IN collect(DISTINCT {usart: u.name, role: ur.role})
                      WHERE x.usart IS NOT NULL] AS usarts
            """,
            {"pin": pin.upper()},
        )
        if not rows:
            return None
        facts = rows[0]
        dev = resolve_device(device) if device else resolve_device("STM32F103RB")
        if device:
            facts["timers"] = [item for item in facts["timers"] if item["timer"] in TIMERS_BY_DEVICE.get(dev.name, ())]
            facts["usarts"] = [item for item in facts["usarts"] if item["usart"] in device_usarts(dev)]
            facts["device_filter"] = dev.name
        return facts

    def timer_facts(self, timer: str) -> dict | None:
        rows = self.run(
            """
            MATCH (t:Timer {name: $timer})
            OPTIONAL MATCH (p:Pin)-[tc:TIM_CH]->(t)
            OPTIONAL MATCH (t)-[dm:DMA]->(dc:DmaChannel)
            OPTIONAL MATCH (t)-[:IRQ]->(i:Irq)
            RETURN t.name AS timer, t.advanced AS advanced, t.bus AS bus,
                   [x IN collect(DISTINCT {channel: tc.channel, pin: p.name}) WHERE x.pin IS NOT NULL] AS channels,
                   [x IN collect(DISTINCT dc.name) WHERE x IS NOT NULL] AS dma,
                   [x IN collect(DISTINCT i.name) WHERE x IS NOT NULL] AS irq
            """,
            {"timer": timer.upper()},
        )
        if not rows:
            return None
        facts = rows[0]
        facts["channels"] = sorted(facts["channels"], key=lambda item: item["channel"])
        return facts

    def usart_facts(self, instance: str) -> dict | None:
        rows = self.run(
            """
            MATCH (u:Usart {name: $name})
            OPTIONAL MATCH (p:Pin)-[ur:USART_ROLE]->(u)
            OPTIONAL MATCH (u)-[dm:DMA]->(dc:DmaChannel)
            OPTIONAL MATCH (u)-[:IRQ]->(i:Irq)
            RETURN u.name AS usart, u.bus AS bus,
                   [x IN collect(DISTINCT {role: ur.role, pin: p.name}) WHERE x.pin IS NOT NULL] AS pins,
                   [x IN collect(DISTINCT {role: dm.role, channel: dc.name}) WHERE x.channel IS NOT NULL] AS dma,
                   [x IN collect(DISTINCT i.name) WHERE x IS NOT NULL] AS irq
            """,
            {"name": instance.upper()},
        )
        if not rows:
            return None
        facts = rows[0]
        facts["pins"] = sorted(facts["pins"], key=lambda item: item["role"])
        facts["dma"] = sorted(facts["dma"], key=lambda item: item["role"])
        return facts

    def pwm_pins(self) -> list[dict]:
        """所有能出 PWM 的引脚（按引脚名排序）。"""
        return self.run(
            """
            MATCH (p:Pin)-[tc:TIM_CH]->(t:Timer)
            RETURN p.name AS pin, t.name AS timer, tc.channel AS channel, t.advanced AS advanced
            ORDER BY p.name, t.name
            """
        )

    def device_facts(self, device: str) -> dict | None:
        rows = self.run(
            """
            MATCH (d:Device {name: $name})
            OPTIONAL MATCH (d)-[:HAS_TIMER]->(t:Timer)
            OPTIONAL MATCH (d)-[:HAS_USART]->(u:Usart)
            RETURN d.name AS device, d.define AS define, d.flash_kb AS flash_kb,
                   d.ram_kb AS ram_kb, d.ports AS ports,
                   [x IN collect(DISTINCT t.name) WHERE x IS NOT NULL] AS timers,
                   [x IN collect(DISTINCT u.name) WHERE x IS NOT NULL] AS usarts
            """,
            {"name": resolve_device(device).name},
        )
        return rows[0] if rows else None

    def verify(self) -> list[str]:
        """从图里读回事实，与 `mcu.py` 现算的结果对比，返回不一致项。"""
        issues: list[str] = []
        for device in DEVICES.values():
            for pin in device_pins(device):
                expected = sorted(f"{timer}:{channel}" for timer, channel in _pin_timer_options(pin, device))
                facts = self.pin_facts(pin.name, device.name)
                got = sorted(
                    f"{item['timer']}:{item['channel']}"
                    for item in (facts or {}).get("timers", [])
                    if item["timer"] in TIMERS_BY_DEVICE.get(device.name, ())
                )
                if expected != got:
                    issues.append(f"{device.name} {pin.name}：表 {expected} ≠ 图 {got}")
        for usart in USART_PINS:
            facts = self.usart_facts(usart) or {}
            pins = {item["role"]: item["pin"] for item in facts.get("pins", [])}
            expected_pins = {role: role_pins[0].name for role, role_pins in USART_PINS[usart].items()}
            if pins != expected_pins:
                issues.append(f"{usart} 引脚：表 {expected_pins} ≠ 图 {pins}")
            dma = {item["role"]: item["channel"] for item in facts.get("dma", [])}
            expected_dma = {role: DMA_MAP[(usart, role)] for role in ("TX", "RX") if (usart, role) in DMA_MAP}
            if dma != expected_dma:
                issues.append(f"{usart} DMA：表 {expected_dma} ≠ 图 {dma}")
        for timer in TIM_CHANNELS:
            facts = self.timer_facts(timer) or {}
            pins = {(item["channel"], item["pin"]) for item in facts.get("channels", [])}
            expected = {(channel, pin_name) for channel, channel_pins in TIM_CHANNELS[timer].items() for pin_name in channel_pins}
            if pins != expected:
                issues.append(f"{timer} 通道：表 {sorted(expected)} ≠ 图 {sorted(pins)}")
        return issues

    # ------------------------------------------------------------ 文本
    def describe(self, kind: str, target: str = "") -> str:
        """把结构化查询结果排版成模型/人读的中文文本。"""
        kind = (kind or "").strip().lower()
        if kind == "pin":
            facts = self.pin_facts(target)
            if facts is None:
                return f"图谱里没有引脚 {target}（F103 只有 PA0~PE15 里的可用引脚）"
            lines = [f"{facts['pin']}（{facts['port']} 口第 {facts['number']} 脚）"]
            if facts["timers"]:
                text = "、".join(
                    f"{item['timer']}_{item['channel']}" + ("（高级定时器）" if item.get("advanced") else "")
                    for item in facts["timers"]
                )
                lines.append(f"- 定时器复用：{text}")
            if facts["usarts"]:
                text = "、".join(f"{item['usart']}_{item['role']}" for item in facts["usarts"])
                lines.append(f"- 串口复用：{text}")
            if len(lines) == 1:
                lines.append("- 没有定时器 / 串口复用，只能当普通 GPIO 用")
            if facts.get("device_filter"):
                lines.append(f"（已按 {facts['device_filter']} 的可用定时器过滤）")
            return "\n".join(lines)

        if kind == "timer":
            facts = self.timer_facts(target)
            if facts is None:
                return f"图谱里没有定时器 {target}"
            lines = [
                f"{facts['timer']}：{'高级定时器' if facts['advanced'] else '通用定时器'}，挂在 {facts['bus']}",
                "- 通道：" + "、".join(f"{item['channel']}→{item['pin']}" for item in facts["channels"]),
            ]
            if facts["dma"]:
                lines.append("- DMA：" + "、".join(facts["dma"]))
            if facts["irq"]:
                lines.append("- 中断：" + "、".join(facts["irq"]))
            return "\n".join(lines)

        if kind == "usart":
            facts = self.usart_facts(target)
            if facts is None:
                return f"图谱里没有串口 {target}"
            pins = {item["role"]: item["pin"] for item in facts["pins"]}
            lines = [
                f"{facts['usart']}：挂在 {facts['bus']}",
                f"- 引脚：TX={pins.get('TX', '?')}，RX={pins.get('RX', '?')}",
            ]
            if facts["dma"]:
                lines.append("- DMA：" + "、".join(f"{item['role']}={item['channel']}" for item in facts["dma"]))
            if facts["irq"]:
                lines.append("- 中断：" + "、".join(facts["irq"]))
            return "\n".join(lines)

        if kind in ("pwm_pins", "pwm"):
            rows = self.pwm_pins()
            lines = [f"可出 PWM 的引脚共 {len(rows)} 个："]
            lines.extend(
                f"- {row['pin']} → {row['timer']}_{row['channel']}" + ("（高级）" if row["advanced"] else "")
                for row in rows
            )
            return "\n".join(lines)

        if kind == "device":
            facts = self.device_facts(target)
            if facts is None:
                return f"图谱里没有器件 {target}"
            timers = sorted(facts["timers"], key=lambda name: (name.startswith("TIM"), name))
            usarts = sorted(facts["usarts"])
            return "\n".join(
                [
                    f"{facts['device']}：{facts['flash_kb']}KB Flash / {facts['ram_kb']}KB RAM，宏 {facts['define']}",
                    f"- 端口：{'、'.join('P' + p for p in facts['ports'])}",
                    f"- 定时器：{'、'.join(timers)}",
                    f"- 串口：{'、'.join(usarts)}",
                ]
            )

        if kind == "stats":
            counts = self.stats()
            lines = ["节点：" + "、".join(f"{k} {v}" for k, v in counts["nodes"].items() if v)]
            lines.append("关系：" + "、".join(f"{k} {v}" for k, v in counts["relationships"].items() if v))
            return "\n".join(lines)

        if kind == "verify":
            issues = self.verify()
            return "图与 mcu.py 完全一致 ✓" if not issues else "\n".join(f"- {item}" for item in issues)

        if kind == "cypher":
            rows = self.run(target, limit=50)
            if not rows:
                return "查询没有返回结果"
            return "\n".join(str(row) for row in rows)

        return f"未知的查询类型：{kind}"


# ---------------------------------------------------------------- 辅助

def _pin_timer_options(pin: Pin, device) -> list[tuple[str, str]]:
    """按 mcu.py 的规则现算（与 `mcu.pin_timer_options` 等价，避免循环导入）。"""
    available = TIMERS_BY_DEVICE.get(device.name, ("TIM1", "TIM2", "TIM3", "TIM4"))
    options: list[tuple[str, str]] = []
    for timer, channels in TIM_CHANNELS.items():
        if timer not in available:
            continue
        for channel, channel_pins in channels.items():
            if pin.name in channel_pins:
                options.append((timer, channel))
    return options


def graph_available(timeout: float = 2.0) -> bool:
    """快速探测 Neo4j 是否可用（Agent 决定要不要挂图谱工具）。"""
    try:
        store = GraphStore(timeout=timeout)
    except RuntimeError:
        return False
    try:
        return store.ping()
    finally:
        store.close()
