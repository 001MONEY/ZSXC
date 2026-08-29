r"""商品注册管理命令行工具。

用法示例：

    注册新商品：
        D:\project\step1\env\python.exe register_goods.py add --sku bag07 --name 新薯片 --type bag --price 3.2

    更新商品（价格/名称等）：
        D:\project\step1\env\python.exe register_goods.py update bag07 --price 4.0
        D:\project\step1\env\python.exe register_goods.py update bag07 --model-class BAG_07_new_snack

    删除商品（软删除，is_active=0）：
        D:\project\step1\env\python.exe register_goods.py delete bag07

    列出商品：
        D:\project\step1\env\python.exe register_goods.py list [--type bag]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from database.goods_dao import GoodsDao  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="智能称重台商品注册管理工具。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add", help="注册新商品")
    add_parser.add_argument("--sku", required=True, help="SKU编码，如 bag07")
    add_parser.add_argument("--name", required=True, help="商品名称")
    add_parser.add_argument("--type", required=True, choices=["bag", "bottle", "box", "cylinder"], help="包装大类")
    add_parser.add_argument("--price", type=float, required=True, help="单价（元）")
    add_parser.add_argument("--model-class", default=None, help="ResNet分类名（如 BAG_07_new_snack），缺省自动生成")
    add_parser.add_argument("--barcode", default=None, help="条码（可选）")
    add_parser.add_argument("--weight-g", type=float, default=None, help="单件参考重量克（可选）")
    add_parser.add_argument("--remark", default=None, help="备注（可选）")

    update_parser = subparsers.add_parser("update", help="更新商品")
    update_parser.add_argument("sku", help="要更新的SKU编码")
    update_parser.add_argument("--name", default=None, help="新商品名称")
    update_parser.add_argument("--type", choices=["bag", "bottle", "box", "cylinder"], default=None, help="新包装大类")
    update_parser.add_argument("--price", type=float, default=None, help="新单价")
    update_parser.add_argument("--model-class", default=None, help="新分类名")
    update_parser.add_argument("--barcode", default=None, help="新条码")
    update_parser.add_argument("--weight-g", type=float, default=None, help="新参考重量克")
    update_parser.add_argument("--remark", default=None, help="新备注")
    update_parser.add_argument("--activate", action="store_true", help="重新启用商品")

    delete_parser = subparsers.add_parser("delete", help="删除商品（默认软删除）")
    delete_parser.add_argument("sku", help="要删除的SKU编码")
    delete_parser.add_argument("--hard", action="store_true", help="物理删除（默认软删除）")

    list_parser = subparsers.add_parser("list", help="列出商品")
    list_parser.add_argument("--type", choices=["bag", "bottle", "box", "cylinder"], default=None, help="按大类过滤")
    list_parser.add_argument("--all", action="store_true", help="包含已停用商品")
    return parser


def cmd_add(args: argparse.Namespace, dao: GoodsDao) -> None:
    ok = dao.add_goods(
        sku_code=args.sku,
        product_name=args.name,
        package_type=args.type,
        unit_price=args.price,
        model_class=args.model_class,
        barcode=args.barcode,
        unit_weight_g=args.weight_g,
        remark=args.remark,
    )
    print(f"✅ 注册成功：{args.sku} {args.name} ¥{args.price:.2f}" if ok else "❌ 注册失败（可能SKU已存在）")


def cmd_update(args: argparse.Namespace, dao: GoodsDao) -> None:
    fields = {}
    if args.name is not None:
        fields["product_name"] = args.name
    if args.type is not None:
        fields["package_type"] = args.type
    if args.price is not None:
        fields["unit_price"] = args.price
    if args.model_class is not None:
        fields["model_class"] = args.model_class
    if args.barcode is not None:
        fields["barcode"] = args.barcode
    if args.weight_g is not None:
        fields["unit_weight_g"] = args.weight_g
    if args.remark is not None:
        fields["remark"] = args.remark
    if args.activate:
        fields["is_active"] = 1
    if not fields:
        print("未提供任何要更新的字段。")
        return
    ok = dao.update_goods(args.sku, **fields)
    print(f"✅ 更新成功：{args.sku} {fields}" if ok else f"❌ 更新失败：{args.sku} 不存在")


def cmd_delete(args: argparse.Namespace, dao: GoodsDao) -> None:
    ok = dao.delete_goods(args.sku, soft=not args.hard)
    mode = "物理删除" if args.hard else "软删除（is_active=0）"
    print(f"✅ {mode}成功：{args.sku}" if ok else f"❌ 删除失败：{args.sku} 不存在")


def cmd_list(args: argparse.Namespace, dao: GoodsDao) -> None:
    if args.type:
        goods = dao.get_by_package_type(args.type)
    else:
        goods = dao.list_all(active_only=not args.all)
    if not goods:
        print("没有商品记录。")
        return
    print(f"{'SKU':<12}{'名称':<16}{'大类':<10}{'单价':>8}  {'分类名'}")
    print("-" * 70)
    for item in goods:
        active = "" if item["is_active"] else " [停用]"
        print(
            f"{item['sku_code']:<12}{item['product_name']:<16}{item['package_type']:<10}"
            f"{float(item['unit_price']):>8.2f}  {item['model_class']}{active}"
        )
    print(f"\n共 {len(goods)} 条")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    dao = GoodsDao()
    try:
        if args.command == "add":
            cmd_add(args, dao)
        elif args.command == "update":
            cmd_update(args, dao)
        elif args.command == "delete":
            cmd_delete(args, dao)
        elif args.command == "list":
            cmd_list(args, dao)
    finally:
        dao.close()


if __name__ == "__main__":
    main()
