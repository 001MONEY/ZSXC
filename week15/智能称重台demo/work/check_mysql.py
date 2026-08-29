"""检查 MySQL 服务与 smart_checkout 数据库状态。"""

import sys

sys.path.insert(0, r"D:\project\step1\week06")
from mysql_db import MySqlHelper

db = MySqlHelper()
ok = db.connect()
print("连接:", ok)
if not ok:
    sys.exit(1)

db.cursor.execute("SHOW DATABASES")
print("数据库列表:", [row["Database"] for row in db.cursor.fetchall()])

db.cursor.execute(
    "SELECT TABLE_NAME FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s",
    (db.db_name,),
)
tables = [row["TABLE_NAME"] for row in db.cursor.fetchall()]
print(f"{db.db_name} 中的表:", tables)

if "products" in tables:
    db.cursor.execute("SHOW CREATE TABLE products")
    print("products 建表语句:", db.cursor.fetchone()["Create Table"])
    db.cursor.execute("SELECT COUNT(*) AS n FROM products")
    print("products 行数:", db.cursor.fetchone()["n"])
    db.cursor.execute("SELECT * FROM products")
    rows = db.cursor.fetchall()
    for row in rows[:10]:
        print("样例:", row)

db.close()
