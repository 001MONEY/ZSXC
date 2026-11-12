# -*- coding: utf-8 -*-
"""Milvus Standalone 连通性 / 基本功能自检脚本（pymilvus 2.6.x）

运行：
    python test_milvus.py
可选参数：
    python test_milvus.py --uri http://127.0.0.1:19530
"""
import argparse
import json
import time
import urllib.request

from pymilvus import DataType, MilvusClient


def check_http_health() -> None:
    """通过 metrics 端口 9091 检查服务端健康状态。"""
    url = "http://127.0.0.1:9091/healthz"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            print(f"[1] HTTP 健康检查 {url} -> {resp.status} {resp.read().decode().strip()}")
    except Exception as exc:  # noqa: BLE001
        print(f"[1] HTTP 健康检查失败（不影响 gRPC 使用）：{exc}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--token", default="")
    parser.add_argument("--collection", default="demo_rag")
    args = parser.parse_args()

    check_http_health()

    client = MilvusClient(uri=args.uri, token=args.token)
    print(f"[2] 已连接 {args.uri}，服务端版本：{client.get_server_version()}")
    print(f"    已有集合：{client.list_collections()}")

    # 重建示例集合：id + 原始文本 + 4 维向量（仅演示）
    if client.has_collection(args.collection):
        client.drop_collection(args.collection)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("text", DataType.VARCHAR, max_length=512)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=4)

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="vector", index_type="AUTOINDEX", metric_type="COSINE")

    client.create_collection(
        collection_name=args.collection,
        schema=schema,
        index_params=index_params,
    )
    print(f"[3] 已创建集合 {args.collection}（AUTOINDEX / COSINE）")

    rows = [
        {"id": 1, "text": "STM32 是常用的 32 位单片机", "vector": [0.9, 0.1, 0.2, 0.0]},
        {"id": 2, "text": "RAG 检索增强生成的基本流程", "vector": [0.1, 0.8, 0.2, 0.1]},
        {"id": 3, "text": "向量数据库用于相似度检索", "vector": [0.2, 0.7, 0.6, 0.0]},
    ]
    client.insert(collection_name=args.collection, data=rows)
    client.flush(args.collection)  # 3.0 下不 flush 时刚写入的数据可能还不可检索
    print(f"[4] 已写入 {len(rows)} 条数据并 flush")

    client.load_collection(args.collection)
    t0 = time.perf_counter()
    results = client.search(
        collection_name=args.collection,
        data=[[0.15, 0.75, 0.3, 0.05]],
        limit=2,
        output_fields=["text"],
        search_params={"metric_type": "COSINE"},
    )
    cost_ms = (time.perf_counter() - t0) * 1000
    print(f"[5] 检索耗时 {cost_ms:.1f} ms，Top2 结果：")
    for hit in results[0]:
        print(f"    id={hit['id']}  score={hit['distance']:.4f}  {hit['entity']['text']}")

    stats = client.get_collection_stats(args.collection)
    print(f"[6] 集合统计：{json.dumps(stats, ensure_ascii=False)}")
    print("\n[OK] Milvus 安装成功，可正常增删查。")


if __name__ == "__main__":
    main()
