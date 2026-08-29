"""商品数据访问层。

提供根据 ResNet 分类输出（model_class）查询商品信息的能力，
供视觉识别流程与 Qt 界面共用。
"""

from __future__ import annotations

from typing import Any

from .mysql_db import MySqlHelper


class GoodsDao:
    """商品表（products）数据访问层。"""

    def __init__(self, helper: MySqlHelper | None = None) -> None:
        self.helper = helper or MySqlHelper()
        self.helper.connect()

    def close(self) -> None:
        self.helper.close()

    def get_by_model_class(self, model_class: str) -> dict[str, Any] | None:
        """根据ResNet分类模型的类别名查询商品（如 'BOTTLE_01_greentea'）。"""
        return self.helper.find_one(
            "SELECT * FROM products WHERE model_class = %s AND is_active = 1",
            (model_class,),
        )

    def get_by_sku_code(self, sku_code: str) -> dict[str, Any] | None:
        """根据业务SKU编码查询商品（如 'bottle01'）。"""
        return self.helper.find_one(
            "SELECT * FROM products WHERE sku_code = %s AND is_active = 1",
            (sku_code,),
        )

    def get_by_package_type(self, package_type: str) -> list[dict[str, Any]]:
        """查询某个包装大类下的全部商品。"""
        return self.helper.find_all(
            "SELECT * FROM products WHERE package_type = %s AND is_active = 1 ORDER BY sku_code",
            (package_type,),
        )

    def list_all(self, active_only: bool = True) -> list[dict[str, Any]]:
        """列出全部商品。"""
        if active_only:
            return self.helper.find_all(
                "SELECT * FROM products WHERE is_active = 1 ORDER BY package_type, sku_code"
            )
        return self.helper.find_all("SELECT * FROM products ORDER BY package_type, sku_code")

    def get_by_barcode(self, barcode: str) -> dict[str, Any] | None:
        """根据条码查询商品。"""
        return self.helper.find_one(
            "SELECT * FROM products WHERE barcode = %s AND is_active = 1",
            (barcode,),
        )

    def add_goods(
        self,
        sku_code: str,
        product_name: str,
        package_type: str,
        unit_price: float,
        model_class: str | None = None,
        barcode: str | None = None,
        unit_weight_g: float | None = None,
        remark: str | None = None,
    ) -> bool:
        """注册新商品。

        model_class 用于对接ResNet分类输出（如 'BAG_07_new_snack'）；
        新商品若尚未训练分类模型，可先填占位值，训练后再更新。
        """
        if model_class is None:
            model_class = f"{package_type.upper()}_{sku_code}"
        affected = self.helper.execute(
            """
            INSERT INTO products
                (sku_code, model_class, product_name, package_type, unit_price, barcode, unit_weight_g, remark)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (sku_code, model_class, product_name, package_type, unit_price, barcode, unit_weight_g, remark),
        )
        return affected > 0

    def update_goods(self, sku_code: str, **fields: Any) -> bool:
        """更新商品信息，仅允许更新白名单字段。

        例如：update_goods('bag07', unit_price=4.0, product_name='新名称')
        """
        allowed = {
            "model_class",
            "product_name",
            "package_type",
            "unit_price",
            "barcode",
            "unit_weight_g",
            "feature_index",
            "remark",
            "is_active",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{key} = %s" for key in updates)
        affected = self.helper.execute(
            f"UPDATE products SET {set_clause} WHERE sku_code = %s",
            (*updates.values(), sku_code),
        )
        return affected > 0

    def delete_goods(self, sku_code: str, soft: bool = True) -> bool:
        """删除商品。soft=True 时置 is_active=0（软删除，保留历史），False 时物理删除。"""
        if soft:
            affected = self.helper.execute(
                "UPDATE products SET is_active = 0 WHERE sku_code = %s",
                (sku_code,),
            )
        else:
            affected = self.helper.execute(
                "DELETE FROM products WHERE sku_code = %s",
                (sku_code,),
            )
        return affected > 0

    def update_by_model_class(self, model_class: str, **fields: Any) -> bool:
        """按 ResNet 分类名（model_class）更新商品，仅允许白名单字段。

        例如：update_by_model_class('BAG_01_kebike_chips', feature_index='lib1_center0')
        """
        allowed = {
            "product_name",
            "package_type",
            "unit_price",
            "barcode",
            "unit_weight_g",
            "feature_index",
            "remark",
            "is_active",
        }
        updates = {key: value for key, value in fields.items() if key in allowed}
        if not updates:
            return False
        set_clause = ", ".join(f"{key} = %s" for key in updates)
        affected = self.helper.execute(
            f"UPDATE products SET {set_clause} WHERE model_class = %s",
            (*updates.values(), model_class),
        )
        return affected > 0

    def summarize(self, items: dict[str, int]) -> dict[str, Any]:
        """汇总识别结果。

        items: {model_class: 数量}，例如 {'BOTTLE_01_greentea': 2}
        返回包含商品明细、总件数、总金额的字典。
        """
        details: list[dict[str, Any]] = []
        total_amount = 0.0
        total_quantity = 0
        for model_class, quantity in items.items():
            goods = self.get_by_model_class(model_class)
            if goods is None:
                details.append(
                    {
                        "model_class": model_class,
                        "quantity": quantity,
                        "found": False,
                        "message": "未注册商品",
                    }
                )
                continue
            amount = float(goods["unit_price"]) * quantity
            total_amount += amount
            total_quantity += quantity
            details.append(
                {
                    "model_class": model_class,
                    "sku_code": goods["sku_code"],
                    "name": goods["product_name"],
                    "package_type": goods["package_type"],
                    "unit_price": float(goods["unit_price"]),
                    "quantity": quantity,
                    "amount": round(amount, 2),
                    "found": True,
                }
            )
        return {
            "details": details,
            "total_quantity": total_quantity,
            "total_amount": round(total_amount, 2),
        }
