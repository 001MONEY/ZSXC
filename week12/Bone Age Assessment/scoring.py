# -*- coding: utf-8 -*-
"""
计分模块：13 根骨头的发育等级 → RUS 得分汇总 → 骨龄查表

流程：grade_dict（rus_id -> 等级）→ 每骨查得分表 → 求和 → 查骨龄表（分男女）

数据驱动设计：所有医学参考数据放在 rus_tables.json
  - score_by_bone : {骨头: {等级: RUS得分}}
  - bone_age_boys  : [[总分, 骨龄年], ...]  （升序，供插值）
  - bone_age_girls : 同上

⚠️⚠️ 重要：rus_tables.json 目前是【占位演示数据】（单调递增的近似值），
    并非官方 RUS-CHN/TW3 参考表！医学上使用前必须替换为官方标准数据。
    替换方式：直接编辑 rus_tables.json（JSON 格式，见生成结构）。

用法：
    python scoring.py --demo                       # 示例等级跑通流程
    python scoring.py --demo --grades 7 9 ...      # 自定义 13 个等级(按 RUS_13 顺序)
"""
import argparse
import json
import sys
from pathlib import Path

import config

TABLES_PATH = Path(__file__).resolve().parent / "rus_tables.json"
CSV_PATH = Path(__file__).resolve().parent / "骨发育等级对照表.csv"

# RUS 13 根骨头（顺序固定，用于参数解析/展示）
RUS_13 = ["Radius", "Ulna", "MCP-1", "MCP-3", "MCP-5",
          "PIP-1", "PIP-3", "PIP-5", "MIP-3", "MIP-5",
          "DIP-1", "DIP-3", "DIP-5"]

# CSV 中文行名 → RUS_13 骨头 id
BONE_NAME_MAP = {"桡骨": "Radius", "尺骨": "Ulna",
                 "掌骨I": "MCP-1", "掌骨III": "MCP-3", "掌骨V": "MCP-5",
                 "近节指骨I": "PIP-1", "近节指骨III": "PIP-3", "近节指骨V": "PIP-5",
                 "中节指骨III": "MIP-3", "中节指骨V": "MIP-5",
                 "远节指骨I": "DIP-1", "远节指骨III": "DIP-3", "远节指骨V": "DIP-5"}

# 每根骨头对应的分类模型关节等级数（决定占位表长度）
GRADE_COUNTS = {"Radius": 14, "Ulna": 12, "MCP-1": 11, "MCP-3": 10, "MCP-5": 10,
                "PIP-1": 12, "PIP-3": 12, "PIP-5": 12, "MIP-3": 12, "MIP-5": 12,
                "DIP-1": 11, "DIP-3": 11, "DIP-5": 11}

# 占位最大得分（仅用于生成演示数据，需替换为官方 RUS-CHN 值）
PLACEHOLDER_MAX = {"Radius": 100, "Ulna": 90, "MCP-1": 70, "MCP-3": 75, "MCP-5": 65,
                   "PIP-1": 60, "PIP-3": 70, "PIP-5": 60, "MIP-3": 60, "MIP-5": 55,
                   "DIP-1": 50, "DIP-3": 65, "DIP-5": 60}
MAX_TOTAL = sum(PLACEHOLDER_MAX.values())   # 占位总分上限（官方为 1000）


# ---------------------------------------------------------------- 数据加载
def build_tables_from_csv(csv_path=None, age_placeholder=True):
    """从官方 RUS-CHN 对照表 CSV 构建计分表。
    csv 格式：第一行 骨发育等级,1,2,...,14；后续每行 骨头名,各等级得分（- 表示无此等级）。
    返回 (tables, max_total)；骨龄表默认仍为占位（需另行提供 RUS总分→骨龄 对照表）。"""
    import csv as _csv
    csv_path = Path(csv_path or CSV_PATH)
    if not csv_path.exists():
        raise FileNotFoundError(f"缺少对照表: {csv_path}")

    rows = list(_csv.reader(csv_path.open(encoding="utf-8-sig")))
    header = rows[0]
    grade_cols = [c for c in header[1:] if c.strip()]          # 等级 1..N
    score_by_bone = {}
    used = set()
    for row in rows[1:]:
        if not row or not row[0].strip():
            continue
        name = row[0].strip()
        if name not in BONE_NAME_MAP:
            continue
        bone = BONE_NAME_MAP[name]
        used.add(bone)
        table = {}
        for i, g in enumerate(grade_cols):
            val = row[i + 1].strip() if i + 1 < len(row) else "-"
            if val not in ("", "-"):
                table[g] = float(val)
        score_by_bone[bone] = table

    missing = [b for b in RUS_13 if b not in score_by_bone]
    if missing:
        raise ValueError(f"CSV 缺少骨头: {missing}")
    max_total = round(sum(max(v.values()) for v in score_by_bone.values()), 1)

    tables = {"_meta": {
        "source": "RUS-CHN 骨发育等级对照表（来自 骨发育等级对照表.csv）",
        "max_total": max_total,
        "bone_age_source": "PLACEHOLDER 占位，需提供 RUS总分→骨龄 官方对照表"},
        "score_by_bone": score_by_bone}
    if age_placeholder:
        # 骨龄表仍为占位（生成与占位表一致的结构）
        placeholder = generate_placeholder_tables()
        tables["bone_age_boys"] = placeholder["bone_age_boys"]
        tables["bone_age_girls"] = placeholder["bone_age_girls"]
    return tables, max_total


def generate_placeholder_tables():
    """生成占位演示表：每骨等级单调递增到最大得分，骨龄表为总分→年 单调插值。
    仅用于演示流程，不可用于医学判断。"""
    score_by_bone = {}
    for bone in RUS_13:
        n = GRADE_COUNTS[bone]
        mx = PLACEHOLDER_MAX[bone]
        # 非线性（前快后慢）近似真实骨成熟曲线
        table = {}
        for g in range(1, n + 1):
            t = (g - 1) / (n - 1)
            table[str(g)] = round(mx * (t ** 1.3), 1)
        score_by_bone[bone] = table

    def make_age_table(male):
        # 占位：总分 0→0.5岁，1000→男17/女16岁，分段线性
        end = 17.0 if male else 16.0
        return [[0, 0.5], [200, 4.0], [400, 8.0], [600, 11.5],
                [800, 14.0], [MAX_TOTAL, end]]

    return {
        "_meta": {"source": "PLACEHOLDER 占位演示数据，非官方RUS-CHN/TW3，医学使用前必须替换",
                  "max_total": MAX_TOTAL},
        "score_by_bone": score_by_bone,
        "bone_age_boys": make_age_table(True),
        "bone_age_girls": make_age_table(False),
    }


def load_tables(path=None, rebuild=False):
    path = path or TABLES_PATH
    if path.exists() and not rebuild:
        return json.loads(path.read_text(encoding="utf-8"))
    # 重建：优先从官方 CSV 构建真实计分表；无 CSV 时用占位表
    if CSV_PATH.exists():
        tables, max_total = build_tables_from_csv(CSV_PATH)
        save_tables(tables, path)
        print(f"[OK] 已从 CSV 构建计分表: {path}（总分上限 {max_total}）")
        return tables
    tables = generate_placeholder_tables()
    save_tables(tables, path)
    print(f"[OK] 未找到 CSV，已生成占位表: {path}")
    return tables


def save_tables(tables, path=None):
    path = path or TABLES_PATH
    path.write_text(json.dumps(tables, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------- 核心逻辑
def grade_to_score(bone, grade, tables):
    """等级 → RUS 得分（等级越界时钳制到合法范围）"""
    table = tables["score_by_bone"][bone]
    grades = sorted(int(k) for k in table)
    if grade is None:
        return None
    g = max(grades[0], min(int(grade), grades[-1]))
    return float(table[str(g)])


def compute_rus_score(grade_dict, tables, sex="boy", require_all=True):
    """
    grade_dict: {rus_id: 等级}，允许缺失部分骨头
    返回: (总得分, 明细list, 缺失骨头list)
    若 require_all 且存在缺失，返回 (None, ...) 表示无法计分。
    """
    missing = [b for b in RUS_13 if b not in grade_dict or grade_dict[b] is None]
    if missing and require_all:
        return None, [], missing

    detail = []
    total = 0.0
    for bone in RUS_13:
        if bone in grade_dict and grade_dict[bone] is not None:
            s = grade_to_score(bone, grade_dict[bone], tables)
            total += s
            detail.append({"bone": bone, "grade": int(grade_dict[bone]), "score": round(s, 1)})
    return round(total, 1), detail, missing


def bone_age_from_rus(total, sex, tables):
    """RUS 总分 → 骨龄（年）。在对照表上线性插值，超出范围钳制到两端。"""
    key = "bone_age_boys" if sex == "boy" else "bone_age_girls"
    table = tables[key]
    if total is None:
        return None
    pts = sorted(table, key=lambda p: p[0])
    if total <= pts[0][0]:
        return pts[0][1]
    if total >= pts[-1][0]:
        return pts[-1][1]
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        if s0 <= total <= s1:
            t = (total - s0) / (s1 - s0) if s1 > s0 else 0
            return round(a0 + t * (a1 - a0), 2)
    return pts[-1][1]


def summarize(grade_dict, sex="boy", require_all=True, tables=None):
    """一步汇总：等级 → 总分 → 骨龄。返回 dict 结果。"""
    tables = tables or load_tables()
    total, detail, missing = compute_rus_score(grade_dict, tables, sex, require_all)
    age = bone_age_from_rus(total, sex, tables) if total is not None else None
    return {"sex": sex, "total_score": total, "bone_age_years": age,
            "detail": detail, "missing": missing}


# ---------------------------------------------------------------- 演示
def demo(grades_args=None):
    tables = load_tables()
    print("=" * 60)
    print(f"计分表来源: {tables['_meta']['source']}")
    print(f"总分上限: {tables['_meta'].get('max_total', MAX_TOTAL)}")

    # 示例等级：取每骨中位等级
    if grades_args:
        assert len(grades_args) == 13, "需要 13 个等级，按 RUS_13 顺序"
        grade_dict = dict(zip(RUS_13, grades_args))
    else:
        grade_dict = {b: max(1, GRADE_COUNTS[b] // 2) for b in RUS_13}

    for sex in ("boy", "girl"):
        res = summarize(grade_dict, sex=sex, tables=tables)
        print(f"\n--- {sex} ---")
        print(f"  总分: {res['total_score']}  骨龄: {res['bone_age_years']} 岁")
        if res["missing"]:
            print(f"  缺失: {res['missing']}")
    print("\n[OK] 演示完成（占位数据，仅供流程验证）")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RUS 计分模块")
    parser.add_argument("--demo", action="store_true", help="跑通计分流程")
    parser.add_argument("--rebuild", action="store_true", help="从 CSV 重新构建计分表")
    parser.add_argument("--grades", nargs="*", type=int, default=None,
                        help="13 个等级（按 Radius,Ulna,MCP-1,... 顺序）")
    args = parser.parse_args()
    if args.rebuild:
        tables, mx = build_tables_from_csv()
        save_tables(tables)
        print(f"[OK] 已重建 rus_tables.json（总分上限 {mx}）")
    elif args.demo:
        demo(args.grades)
    else:
        print("请用 --demo 运行演示，或用 --rebuild 重建计分表")
