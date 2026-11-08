"""在已导入的 MySQL 上核对作业验收题和两个失败回路。"""
import argparse
from collections import Counter
from dataclasses import replace
from pathlib import Path

from dotenv import load_dotenv
from pymysql.err import OperationalError
from sqlglot.errors import SqlglotError

from agent import (
    COUNT_SQL, QUESTION, MySQLReader, QueryResult, ScriptedPlanner,
    build_graph, result_error, validate_sql,
)
from db import connect


class OneBadResult(MySQLReader):
    def __init__(self):
        self.calls = 0

    def explain_and_execute(self, sql):
        result = super().explain_and_execute(sql)
        self.calls += 1
        # SQL 真执行。只在第一次查询后注入空结果以验证结果校验分支。
        return replace(result, rows=[]) if self.calls == 1 else result


class AlwaysBadDraft(ScriptedPlanner):
    def generate(self, state):
        return "SELECT COUNT(*) FROM employees WHERE nonexistent >= '1988-01-01'"


def verify():
    reader = MySQLReader()
    with connect() as connection, connection.cursor() as cursor:
        cursor.execute(COUNT_SQL)
        baseline = cursor.fetchone()[0]
        assert baseline == 195057, baseline
        try:
            cursor.execute("UPDATE employees SET hire_date = hire_date WHERE 1 = 0")
        except OperationalError as error:
            assert error.args[0] in (1142, 1143), error.args[0]
        else:
            raise AssertionError("查询账号有 UPDATE 权限")
    print("MySQL independent baseline: 195057, reader UPDATE denied")

    schema = reader.schema(["employees"])
    adversarial = {
        "多语句": COUNT_SQL + "; SELECT * FROM employees",
        "写操作": "DELETE FROM employees",
        "跨库": "SELECT COUNT(*) FROM mysql.user",
        "导出": "SELECT emp_no FROM employees INTO OUTFILE 'blocked.txt'",
        "休眠函数": "SELECT SLEEP(1) FROM employees",
        "未知列": COUNT_SQL.replace("hire_date", "unknown_column"),
        "错误日期": COUNT_SQL.replace("1988-01-01", "1987-01-01"),
    }
    for name, draft in adversarial.items():
        try:
            validate_sql(draft, schema, "count", employee_1987=True)
        except (ValueError, SqlglotError):
            continue
        raise AssertionError(f"校验未拦截：{name}")
    assert result_error(QueryResult(["n"], [(0,)], [], False), "count") == ""
    print("SQL guard: 7 invalid drafts denied, COUNT zero accepted")

    injected_reader = OneBadResult()
    graph = build_graph(injected_reader, ScriptedPlanner(bad_first_draft=True))
    final, path, errors = {}, [], []
    for chunk in graph.stream({"question": QUESTION}, stream_mode="updates", config={"recursion_limit": 32}):
        for node, update in chunk.items():
            path.append(node)
            final.update(update)
            if update.get("failure_stage"):
                errors.append(update["failure_stage"])
    assert errors == ["SQL 校验", "结果校验"], errors
    counts = Counter(path)
    assert counts["draft_sql"] == 3 and counts["execute_sql"] == 2
    assert counts["check_result"] == 2 and counts["recognize_intent"] == counts["search_schema"] == 1
    assert injected_reader.calls == 2
    assert final["status"] == "success" and final["result"].rows == [(baseline,)]
    print("回路:", " → ".join(path))
    print("第 1 次错列未执行，第 2 次注入异常结果，第 3 次返回", baseline)

    failed = build_graph(reader, AlwaysBadDraft(), max_attempts=2).invoke({"question": QUESTION})
    assert failed["status"] == "failed" and failed["attempt"] == 2 and "result" not in failed
    rejected = build_graph(reader, ScriptedPlanner()).invoke({"question": "删除员工表"})
    assert rejected["status"] == "rejected" and "schema" not in rejected
    print("重试耗尽后停止，拒绝写请求，验证通过")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    load_dotenv(args.env_file)
    verify()
