"""自建 MCP Server：校园图书馆（Resource + Tool 分工示范）。

课堂里由学生自己起服务；这里给 notebook 用的版本，默认走 stdio（由客户端拉起子进程），
也支持 SSE：python mcp_library_server.py --transport sse --port 8000

注意：mcp 1.x 里类名是 FastMCP（课堂笔记里的 mcp.server.mcpserver.MCPServer 在旧/新版中命名不同）。
"""

from __future__ import annotations

import argparse

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("library")

STUDENTS = {
    "2024001": {"name": "张三", "borrowed": 2, "overdue": 0},
    "2024002": {"name": "李四", "borrowed": 5, "overdue": 1},
}

STOCK = {
    "三体": 3,
    "红楼梦": 0,
    "Python入门": 5,
}


# ---------- Resource：只读上下文，注入 prompt ----------
@mcp.resource("library://rules")
def library_rules() -> str:
    """图书馆借阅规则。"""
    return (
        "1. 每人最多借 5 本\n"
        "2. 借期 30 天，可续借 1 次\n"
        "3. 逾期每天 0.5 元\n"
        "4. 有逾期未还，禁止继续借书\n"
        "5. 参考书不可外借"
    )


@mcp.resource("student://{student_id}")
def student_profile(student_id: str) -> str:
    """按学号读取学生借阅档案。"""
    info = STUDENTS.get(student_id)
    if not info:
        return f"未找到学号 {student_id}"
    return (
        f"学号:{student_id}, 姓名:{info['name']}, "
        f"已借:{info['borrowed']}本, 逾期:{info['overdue']}本"
    )


# ---------- Tool：可执行动作，由模型调用 ----------
@mcp.tool()
def check_stock(book_name: str) -> str:
    """查询某本书当前可借数量。"""
    return f"《{book_name}》可借数量：{STOCK.get(book_name, 0)}"


@mcp.tool()
def borrow_book(student_id: str, book_name: str) -> str:
    """为学生办理借书。"""
    student = STUDENTS.get(student_id)
    if not student:
        return f"失败：学号 {student_id} 不存在"
    if student["overdue"] > 0:
        return f"失败：{student['name']} 有逾期图书，禁止借阅"
    if student["borrowed"] >= 5:
        return f"失败：{student['name']} 已借满 5 本"

    stock = STOCK.get(book_name, 0)
    if stock <= 0:
        return f"失败：《{book_name}》暂无库存"

    STOCK[book_name] = stock - 1
    student["borrowed"] += 1
    return (
        f"成功：{student['name']} 借到《{book_name}》，"
        f"当前已借 {student['borrowed']} 本，剩余库存 {STOCK[book_name]} 本"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", default="stdio", choices=["stdio", "sse"])
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.transport == "sse":
        # 本机 + 局域网可访问；SSE 需要把 host/port 传进构造函数
        mcp.settings.host = "0.0.0.0"
        mcp.settings.port = args.port
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
