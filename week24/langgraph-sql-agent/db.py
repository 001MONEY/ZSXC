import os
import re

import pymysql


TABLES = ("employees", "departments", "dept_emp", "dept_manager", "salaries", "titles")


def database_name():
    name = os.environ.get("MYSQL_DATABASE", "47")
    if not re.fullmatch(r"[A-Za-z0-9_]{1,64}", name):
        raise ValueError("MYSQL_DATABASE 必须由字母、数字或下划线组成")
    return name


def connect(*, admin=False, select_database=True):
    if admin:
        user = os.environ.get("MYSQL_ADMIN_USER", "root")
        password = os.environ["MYSQL_ROOT_PASSWORD"]
    else:
        user = os.environ.get("SQL_AGENT_USER", "sql_agent_reader")
        password = os.environ["SQL_AGENT_PASSWORD"]
        if user.lower() in {"root", os.environ.get("MYSQL_ADMIN_USER", "root").lower()}:
            raise ValueError("Agent 必须使用独立的只读账号")
    return pymysql.connect(
        host=os.environ.get("MYSQL_HOST", "127.0.0.1"),
        port=int(os.environ.get("MYSQL_PORT", "13307")),
        user=user,
        password=password,
        database=database_name() if select_database else None,
        charset="utf8mb4",
        autocommit=True,
        connect_timeout=5,
        read_timeout=15 if not admin else 120,
        write_timeout=15 if not admin else 120,
    )
