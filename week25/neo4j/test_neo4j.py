"""Neo4j 连通性 / 基础 Cypher 自测脚本。

用法（在 D:\\project\\step1\\week25 下）：
    d:\\project\\step1\\week23\\.venv\\Scripts\\python.exe neo4j\\test_neo4j.py

依赖：pip/uv 安装 neo4j 驱动（>=5.x）
"""

from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "neo4j123456")


def main() -> None:
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()

    with driver.session() as session:
        # 服务器信息
        info = session.run(
            "CALL dbms.components() YIELD name, versions, edition "
            "RETURN name, versions[0] AS version, edition"
        ).single()
        print(f"[components] {info['name']} {info['version']} ({info['edition']})")

        # APOC 是否可用
        try:
            apoc = session.run(
                "RETURN apoc.version() AS v"
            ).single()
            print(f"[apoc] {apoc['v']}")
        except Exception as exc:  # noqa: BLE001
            print(f"[apoc] 不可用: {exc}")

        # 写入 + 查询 + 清理
        session.run(
            """
            MERGE (a:Person {name: 'Alice'})
            MERGE (b:Person {name: 'Bob'})
            MERGE (a)-[:KNOWS {since: 2024}]->(b)
            """
        )
        rows = session.run(
            """
            MATCH (a:Person)-[r:KNOWS]->(b:Person)
            RETURN a.name AS src, b.name AS dst, r.since AS since
            """
        )
        for row in rows:
            print(f"[graph] {row['src']} -[:KNOWS {row['since']}]-> {row['dst']}")

        # 向量索引能力（Neo4j 5.11+ 社区版支持）
        session.run(
            """
            CREATE VECTOR INDEX person_embedding IF NOT EXISTS
            FOR (p:Person) ON (p.embedding)
            OPTIONS {indexConfig: {
                `vector.dimensions`: 8,
                `vector.similarity_function`: 'cosine'
            }}
            """
        )
        idx = session.run(
            "SHOW INDEXES YIELD name, type, state "
            "WHERE name = 'person_embedding' RETURN name, type, state"
        ).single()
        print(f"[vector-index] {dict(idx)}")

        session.run("MATCH (p:Person) WHERE p.name IN ['Alice','Bob'] DETACH DELETE p")
        print("[cleanup] done, nodes removed")

    driver.close()
    print("OK: Neo4j 连接与读写正常")


if __name__ == "__main__":
    main()
