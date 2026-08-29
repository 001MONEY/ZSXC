"""MySQL 数据库操作辅助类（复用 week06 的 MySqlHelper）。

保持项目自包含，无需依赖 week06 目录即可使用。

连接参数支持环境变量覆盖（Docker 容器内连宿主机 MySQL 用）：
  MYSQL_HOST / MYSQL_PORT / MYSQL_USER / MYSQL_PWD / MYSQL_DB
宿主机不设置时使用默认值 localhost:3306 root/8888 smart_checkout。
"""

import os

import pymysql
from pymysql.err import OperationalError
from pymysql.cursors import DictCursor


class MySqlHelper:
    """MySQL 数据库操作辅助类"""

    def __init__(self, host=None, port=None, user=None, pwd=None, db_name=None):
        self.host = host or os.environ.get("MYSQL_HOST", "localhost")
        self.port = int(port or os.environ.get("MYSQL_PORT", "3306"))
        self.user = user or os.environ.get("MYSQL_USER", "root")
        self.pwd = pwd or os.environ.get("MYSQL_PWD", "8888")
        self.db_name = db_name or os.environ.get("MYSQL_DB", "smart_checkout")
        self.connection = None
        self.cursor = None

    def connect(self):
        """建立数据库连接"""
        try:
            self.connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.pwd,
                database=self.db_name,
                charset="utf8mb4",
                cursorclass=DictCursor,
            )
            self.cursor = self.connection.cursor()
            return True
        except OperationalError as e:
            code, msg = e.args
            print(f"❌ 连接失败 (错误码 {code}): {msg}")
            return False

    def close(self):
        """关闭连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def find_all(self, sql, params=None):
        """查询多条数据，返回字典列表"""
        if not self.connection or not self.connection.open:
            self.connect()
        self.cursor.execute(sql, params or ())
        return self.cursor.fetchall()

    def find_one(self, sql, params=None):
        """查询单条数据，返回字典"""
        result = self.find_all(sql, params)
        return result[0] if result else None

    def execute(self, sql, params=None):
        """执行插入/更新/删除，返回受影响行数"""
        if not self.connection or not self.connection.open:
            self.connect()
        affected = self.cursor.execute(sql, params or ())
        self.connection.commit()
        return affected
