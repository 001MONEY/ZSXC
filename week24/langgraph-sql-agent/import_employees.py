"""将课堂 Navicat SQL 导入空库，合并逐行 INSERT 以减少提交次数。"""
import argparse
from collections import Counter
from pathlib import Path
import os
import re

from dotenv import load_dotenv

from db import TABLES, connect, database_name


INSERT = re.compile(r"INSERT INTO `([a-z_]+)` VALUES (\(.+\));")
DDL = re.compile(r"(?:CREATE TABLE|DROP TABLE IF EXISTS) `([a-z_]+)`")
SETTINGS = {"SET NAMES utf8mb4;", "SET FOREIGN_KEY_CHECKS = 0;", "SET FOREIGN_KEY_CHECKS = 1;"}


def import_dump(cursor, path):
    counts = Counter()
    table, values, ddl, in_header = None, [], [], False

    def flush():
        if values:
            cursor.execute(f"INSERT INTO `{table}` VALUES " + ",".join(values))
            counts[table] += len(values)
            values.clear()

    with path.open(encoding="utf-8-sig") as stream:
        for line_no, raw in enumerate(stream, 1):
            line = raw.strip()
            if line.startswith("/*") and not line.startswith("/*!"):
                in_header = not line.endswith("*/")
                continue
            if in_header:
                in_header = not line.endswith("*/")
                continue
            if not line or line.startswith("--"):
                continue
            record = INSERT.fullmatch(line)
            if record:
                target, row = record.groups()
                if target not in TABLES or ddl:
                    raise ValueError(f"第 {line_no} 行 INSERT 不符合课堂数据结构")
                if target != table or len(values) >= 1500:
                    flush()
                    table = target
                values.append(row)
                continue
            flush()
            if ddl:
                ddl.append(line)
                if line.endswith(";"):
                    cursor.execute("\n".join(ddl))
                    ddl.clear()
                continue
            if line in SETTINGS:
                cursor.execute(line)
                continue
            match = DDL.match(line)
            if not match or match[1] not in TABLES:
                raise ValueError(f"第 {line_no} 行不是受支持的课堂导出语句")
            if line.startswith("CREATE TABLE") and not line.endswith(";"):
                ddl.append(line)
            else:
                cursor.execute(line)
    flush()
    if ddl or in_header or set(counts) != set(TABLES):
        raise ValueError("导出文件不完整，或与课堂六表结构不一致")
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    args = parser.parse_args()
    if not args.dump.is_file():
        parser.error("SQL 文件不存在")
    load_dotenv(args.env_file)
    db = database_name()
    reader = os.environ.get("SQL_AGENT_USER", "sql_agent_reader")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,32}", reader) or reader.lower() == "root":
        raise ValueError("必须配置单独的 SQL_AGENT_USER")
    reader_password = os.environ["SQL_AGENT_PASSWORD"]
    if not reader_password:
        raise ValueError("SQL_AGENT_PASSWORD 不能为空")
    with connect(admin=True, select_database=False) as connection, connection.cursor() as cur:
        cur.execute(f"CREATE DATABASE IF NOT EXISTS `{db}` CHARACTER SET utf8mb4")
        connection.select_db(db)
        cur.execute("SHOW FULL TABLES")
        if cur.fetchall():
            raise SystemExit("目标库已有表。为保留现有数据，请为本次导入指定一个新库名。")
        cur.execute("SELECT COUNT(*) FROM mysql.user WHERE User = %s", (reader,))
        if cur.fetchone()[0]:
            raise SystemExit("查询账号已存在，请配置一个新的 SQL_AGENT_USER。")
        counts = import_dump(cur, args.dump)
        cur.execute("CREATE USER %s@'%%' IDENTIFIED BY %s", (reader, reader_password))
        for table in TABLES:
            cur.execute(f"GRANT SELECT ON `{db}`.`{table}` TO %s@'%%'", (reader,))
        for table in TABLES:
            print(f"{table}: {counts[table]:,} rows imported")
        cur.execute("SELECT COUNT(*) FROM employees WHERE hire_date >= '1988-01-01'")
        print(f"MySQL baseline for hire_date >= 1988-01-01: {cur.fetchone()[0]:,}")


if __name__ == "__main__":
    main()
