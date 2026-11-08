import argparse
from dataclasses import dataclass
import importlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from pymysql import MySQLError
from sqlglot import exp, parse
from sqlglot.errors import SqlglotError
from sqlglot.optimizer.qualify import qualify

from db import TABLES, connect, database_name


QUESTION = "1987 年之后入职的员工人数是多少？"
COUNT_SQL = "SELECT COUNT(*) AS employee_count FROM employees WHERE hire_date >= '1988-01-01'"
MAX_ROWS = 50
TABLE_HINTS = {
    "employees": "员工基本信息，入职日期、姓名、性别",
    "departments": "部门名称、部门编号",
    "dept_emp": "员工与部门的历史关联",
    "dept_manager": "部门经理和任职区间",
    "salaries": "员工薪资及生效区间",
    "titles": "员工职称及任职区间",
}
SAFE_FUNCTIONS = {
    "COUNT", "SUM", "AVG", "MIN", "MAX", "ROUND", "ABS",
    "COALESCE", "NULLIF", "LOWER", "UPPER", "CONCAT",
    "YEAR", "MONTH", "DAY", "DATE", "CAST", "CURRENT_DATE", "IF", "BETWEEN",
    "AND", "OR", "NOT", "CASE",
}


class Intent(BaseModel):
    allowed: bool = Field(description="是否是本数据集可回答的只读查询")
    shape: Literal["count", "table"] = Field(description="单个总人数/总条数为 count，其余为 table")
    tables: list[str] = Field(description="最多四张相关表，使用表的真实名称")


class Draft(BaseModel):
    sql: str = Field(description="一条 MySQL SELECT 语句，不包含 Markdown 或解释")


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[tuple]
    plan: list[dict]
    truncated: bool


class AgentState(TypedDict, total=False):
    question: str
    intent: dict
    schema: dict
    employee_1987: bool
    attempt: int
    draft: str
    sql: str
    error: str
    status: Literal["running", "success", "rejected", "failed"]
    result: QueryResult
    answer: str
    failure_stage: str


class MySQLReader:
    def schema(self, tables):
        schema = {table: {} for table in tables}
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute(
                "SELECT TABLE_NAME, COLUMN_NAME, DATA_TYPE, COLUMN_KEY, COLUMN_COMMENT "
                "FROM information_schema.COLUMNS WHERE TABLE_SCHEMA = %s "
                "ORDER BY TABLE_NAME, ORDINAL_POSITION",
                (database_name(),),
            )
            for table, column, datatype, key, comment in cursor.fetchall():
                if table in schema:
                    schema[table][column] = {"type": datatype, "key": key, "comment": comment}
        return {table: fields for table, fields in schema.items() if fields}

    def explain_and_execute(self, sql):
        with connect() as connection, connection.cursor() as cursor:
            cursor.execute("SET SESSION MAX_EXECUTION_TIME = 5000")
            cursor.execute("START TRANSACTION READ ONLY")
            cursor.execute("EXPLAIN " + sql)
            plan_names = [item[0] for item in cursor.description]
            plan = [dict(zip(plan_names, row)) for row in cursor.fetchall()]
            cursor.execute(sql)
            columns = [item[0] for item in cursor.description]
            rows = list(cursor.fetchmany(MAX_ROWS + 1))
            connection.rollback()
        return QueryResult(columns, rows[:MAX_ROWS], plan, len(rows) > MAX_ROWS)


def is_verification_question(question):
    compact = re.sub(r"\s+", "", question)
    return bool(re.fullmatch(
        r"(?:请(?:查询|统计|问))?1987年(?:之|以)?后(?:入职|加入)的?员工(?:总)?人数"
        r"(?:是多少|有多少)?[？?。]?", compact,
    ))


class ScriptedPlanner:
    """离线演示使用固定草稿和意图，只支持课堂验收问题。"""

    def __init__(self, bad_first_draft=False):
        self.bad_first_draft = bad_first_draft

    def classify(self, question):
        return Intent(allowed=is_verification_question(question), shape="count", tables=["employees"])

    def generate(self, state):
        if self.bad_first_draft and state["attempt"] == 1:
            return COUNT_SQL.replace("hire_date", "join_date")
        return COUNT_SQL


class ModelPlanner:
    def __init__(self, model):
        self.intent_model = model.with_structured_output(Intent, method="function_calling")
        self.draft_model = model.with_structured_output(Draft, method="function_calling")

    def classify(self, question):
        return self.intent_model.invoke([
            SystemMessage(content=(
                "识别员工数据库的查询意图。只接受可由现有六表回答的只读查询，"
                "拒绝数据修改、数据库管理和无关问题。"
                "单个总人数/总条数使用 count，分组统计使用 table。"
                f"表的用途：{json.dumps(TABLE_HINTS, ensure_ascii=False)}。"
            )),
            HumanMessage(content=question),
        ])

    def generate(self, state):
        payload = {
            "question": state["question"],
            "shape": state["intent"]["shape"],
            "schema": state["schema"],
            "previous_sql": state.get("draft", ""),
            "previous_validation_error": state.get("error", ""),
        }
        prompt = (
            "根据用户问题和 schema 生成一条 MySQL SELECT。仅用给定的表和字段，不加数据库名。"
            "支持常见过滤、JOIN、GROUP BY、排序与聚合。不用子查询、CTE、UNION、变量或注释。"
            "单个总人数使用 COUNT(*) 并命名返回列。员工表 emp_no 是唯一员工编号。"
            "连接员工相关表时通常使用 emp_no，部门相关表使用 dept_no。"
            "按自然年理解“1987 年之后”，从 1988-01-01（含）起算。"
            "收到前一次错误时修复草稿。"
        )
        draft = self.draft_model.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
        ])
        return draft.sql


def _unwrap(node):
    while isinstance(node, exp.Paren):
        node = node.this
    return node


def _verify_employee_count(statement, tables, count):
    count_arg = count.this
    valid_count = isinstance(count_arg, exp.Star) or (
        isinstance(count_arg, exp.Column) and count_arg.name.lower() == "emp_no"
    )
    forbidden_clauses = ("joins", "group", "having", "qualify", "offset")
    if tables != ["employees"] or not valid_count or any(statement.args.get(k) for k in forbidden_clauses):
        raise ValueError("验收题需要从 employees 单表统计员工总数")
    where = statement.args.get("where")
    condition = _unwrap(where.this) if where else None
    for operator, boundary in ((exp.GTE, "1988-01-01"), (exp.GT, "1987-12-31")):
        if (
            isinstance(condition, operator)
            and isinstance(condition.left, exp.Column)
            and condition.left.name.lower() == "hire_date"
            and isinstance(condition.right, exp.Literal)
            and condition.right.is_string
            and condition.right.this == boundary
        ):
            return
    raise ValueError("验收题请使用 hire_date >= '1988-01-01'，含当天")


def validate_sql(draft, schema, shape, employee_1987=False):
    if len(draft) > 4000 or re.search(r"--|/\*|#|\x00", draft):
        raise ValueError("SQL 超长或包含注释")
    statements = parse(draft, read="mysql")
    if len(statements) != 1 or not isinstance(statements[0], exp.Select):
        raise ValueError("只允许一条 SELECT")
    statement = statements[0]
    forbidden = (exp.CTE, exp.Subquery, exp.SetOperation, exp.Into, exp.Lock,
                 exp.Parameter, exp.SessionParameter, exp.PropertyEQ, exp.Dot, exp.Window)
    if any(isinstance(node, forbidden) for node in statement.walk()):
        raise ValueError("不支持子查询、CTE、集合操作、变量、窗口函数或带副作用的子句")
    tables = []
    for table in statement.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier) or table.db or table.catalog or table.name not in schema:
            raise ValueError("SQL 引用了未检索到的表或数据库")
        tables.append(table.name)
    if not tables:
        raise ValueError("SELECT 必须引用允许的课堂表")
    for node in statement.find_all(exp.Func):
        if node.sql_name().upper() not in SAFE_FUNCTIONS:
            raise ValueError(f"不允许使用函数 {node.sql_name()}")
    if shape == "count":
        fields = statement.expressions
        if len(fields) != 1 or not isinstance(fields[0].unalias(), exp.Count) or statement.args.get("group"):
            raise ValueError("总数查询必须返回单个 COUNT 聚合")
        if employee_1987:
            _verify_employee_count(statement, tables, fields[0].unalias())
    else:
        limit = statement.args.get("limit")
        if not limit or not isinstance(limit.expression, exp.Literal) or not limit.expression.is_int:
            statement.limit(MAX_ROWS + 1, copy=False)
        elif not 1 <= int(limit.expression.this) <= MAX_ROWS + 1:
            statement.limit(MAX_ROWS + 1, copy=False)
    # 在 MySQL EXPLAIN 前检测不存在、无法消歧的字段。
    typed_schema = {name: {col: props["type"] for col, props in columns.items()}
                    for name, columns in schema.items()}
    qualified = qualify(statement, dialect="mysql", schema=typed_schema, identify=True)
    return qualified.sql(dialect="mysql")


def result_error(result, shape):
    if not result.columns or any(len(row) != len(result.columns) for row in result.rows):
        return "返回的列与行结构不一致"
    if shape == "count":
        if (
            len(result.columns) != 1 or len(result.rows) != 1 or result.truncated
            or type(result.rows[0][0]) is not int or result.rows[0][0] < 0
        ):
            return "总数查询必须返回单行单列非负整数，0 也是有效结果"
    return ""


def build_graph(reader, planner, max_attempts=3):
    if not 1 <= max_attempts <= 5:
        raise ValueError("max_attempts 必须在 1～5 之间")

    def recognize_intent(state):
        question = state["question"].strip()
        if not question or len(question) > 1000:
            return {"status": "rejected", "error": "请输入 1～1000 字的员工数据查询问题"}
        intent = planner.classify(question)
        if not intent.allowed:
            return {"status": "rejected", "error": "无法处理该问题，请提出员工库内的只读查询"}
        known_count = is_verification_question(question)
        if known_count:
            intent = Intent(allowed=True, shape="count", tables=["employees"])
        return {"intent": intent.model_dump(), "employee_1987": known_count, "status": "running"}

    def search_schema(state):
        candidates = list(dict.fromkeys(t for t in state["intent"]["tables"] if t in TABLES))[:4]
        if state["employee_1987"]:
            candidates = ["employees"]
        if not candidates:
            candidates = ["employees"]
        schema = reader.schema(candidates)
        if not schema:
            return {"status": "failed", "error": "未找到相关表，请确认数据库已完成导入"}
        return {"schema": schema}

    def draft_sql(state):
        attempt = state.get("attempt", 0) + 1
        draft = planner.generate({**state, "attempt": attempt})
        return {"attempt": attempt, "draft": draft, "sql": "", "error": "", "failure_stage": ""}

    def check_sql(state):
        try:
            sql = validate_sql(state["draft"], state["schema"], state["intent"]["shape"], state["employee_1987"])
        except (SqlglotError, ValueError) as exc:
            return {"failure_stage": "SQL 校验", "error": str(exc)[:240]}
        return {"sql": sql}

    def execute_sql(state):
        try:
            result = reader.explain_and_execute(state["sql"])
        except MySQLError as exc:
            code = exc.args[0] if exc.args and isinstance(exc.args[0], int) else 0
            return {"failure_stage": "数据库执行", "error": f"MySQL 错误 {code}，请检查语法、字段或查询范围"}
        return {"result": result}

    def check_result(state):
        error = result_error(state["result"], state["intent"]["shape"])
        if error:
            return {"failure_stage": "结果校验", "error": error}
        return {"status": "success"}

    def answer(state):
        if state["status"] == "rejected":
            return {"answer": state["error"]}
        if state.get("status") != "success":
            prefix = f"尝试 {state.get('attempt', 0)} 次后未能完成查询"
            return {"status": "failed", "answer": f"{prefix}。{state.get('error', '')}"}
        result = state["result"]
        if state["employee_1987"]:
            return {"answer": f"1987 年之后（含 1988-01-01 当天）入职的员工共有 {result.rows[0][0]:,} 人。"}
        if state["intent"]["shape"] == "count":
            return {"answer": f"符合查询条件的总数为 {result.rows[0][0]:,}。"}
        if not result.rows:
            return {"answer": "没有查到符合条件的记录。"}
        suffix = "，已截断" if result.truncated else ""
        return {"answer": f"查询返回 {len(result.rows)} 条记录{suffix}。"}

    def repair_or_finish(state, next_node):
        if not state.get("error"):
            return next_node
        return "draft_sql" if state["attempt"] < max_attempts else "answer"

    graph = StateGraph(AgentState)
    for node in (recognize_intent, search_schema, draft_sql, check_sql, execute_sql, check_result, answer):
        graph.add_node(node)
    graph.add_edge(START, "recognize_intent")
    graph.add_conditional_edges("recognize_intent", lambda s: "answer" if s.get("error") else "search_schema",
                                ["answer", "search_schema"])
    graph.add_conditional_edges("search_schema", lambda s: "answer" if s.get("error") else "draft_sql",
                                ["answer", "draft_sql"])
    graph.add_edge("draft_sql", "check_sql")
    graph.add_conditional_edges("check_sql", lambda s: repair_or_finish(s, "execute_sql"),
                                ["draft_sql", "execute_sql", "answer"])
    graph.add_conditional_edges("execute_sql", lambda s: repair_or_finish(s, "check_result"),
                                ["draft_sql", "check_result", "answer"])
    graph.add_conditional_edges("check_result", lambda s: repair_or_finish(s, "answer"), ["draft_sql", "answer"])
    graph.add_edge("answer", END)
    return graph.compile()


def load_course_model(lab_directory):
    path = lab_directory.resolve()
    if not (path / "lab_config.py").is_file():
        raise ValueError("课程配置目录应包含 lab_config.py")
    sys.path.insert(0, str(path))
    config = importlib.import_module("lab_config")
    return config.get_model(temperature=0, timeout=25, max_retries=1)


def main():
    parser = argparse.ArgumentParser(description="LangGraph MySQL 员工库查询作业")
    parser.add_argument("question", nargs="?", default=QUESTION)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--lab-dir", type=Path, help="课程 lab_config.py 所在目录；提供后启用真模型")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    load_dotenv(args.env_file)
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    planner = ModelPlanner(load_course_model(args.lab_dir)) if args.lab_dir else ScriptedPlanner()
    graph = build_graph(MySQLReader(), planner, args.max_attempts)
    final = {}
    for chunk in graph.stream({"question": args.question, "status": "running", "attempt": 0},
                              config={"recursion_limit": 32}, stream_mode="updates"):
        for node, update in chunk.items():
            print(f"[{node}]")
            if update.get("error"):
                print("  " + update["error"])
            if update.get("sql"):
                print("  " + update["sql"])
            if "result" in update:
                print("  EXPLAIN:", json.dumps(update["result"].plan, default=str, ensure_ascii=False))
            final.update(update)
    print(final["answer"])
    if final.get("status") == "success":
        result = final["result"]
        print(json.dumps({"columns": result.columns, "rows": result.rows}, default=str, ensure_ascii=False))
    return 0 if final.get("status") == "success" else 2


if __name__ == "__main__":
    sys.exit(main())
