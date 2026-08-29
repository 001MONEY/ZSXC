r"""智能称重台 Qt 展示界面（PySide6）。

用法：
  D:\project\step1\env\python.exe main.py

功能：
  - 输入源切换：摄像头（0/1）或本地视频文件
  - 实时显示带检测框/商品名的画面，右侧购物车明细 + 合计金额
  - 未注册商品红色提示（不会计入购物车）
  - 结算弹窗、购物车重置
推理复用冻结的 ONNX 引擎（onnx_engine.py），阈值 sim>=0.80 / margin>=0.15。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from inference_worker import InferenceWorker  # noqa: E402
from database.goods_dao import GoodsDao  # noqa: E402
from feature_library_updater import register_sku, suggest_model_class, suggest_sku  # noqa: E402

GROUP_LABELS = {"bag": "袋装", "bottle": "瓶装", "box": "盒装", "cylinder": "罐装"}


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("智能称重台系统")
        self.resize(1360, 820)

        self.worker = InferenceWorker(self)
        self.worker.frame_ready.connect(self.on_frame_ready)
        self.worker.error.connect(self.on_worker_error)
        self.worker.input_ended.connect(self.on_input_ended)

        self._last_cart: dict | None = None
        self._last_unknown: str = ""
        self._last_unknown_group: str | None = None
        self._selected_video: str | None = None
        self._build_ui()
        self._set_running_state(False)

    # ------------------------------------------------------------ UI 构建
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        # 顶部控制条
        top = QHBoxLayout()
        title = QLabel("智能称重台系统")
        title.setStyleSheet("font-size:22px; font-weight:bold; color:#1a5;")
        top.addWidget(title)
        top.addSpacing(20)

        top.addWidget(QLabel("输入源:"))
        self.source_combo = QComboBox()
        self.source_combo.addItem("摄像头 0", 0)
        self.source_combo.addItem("摄像头 1", 1)
        self.source_combo.addItem("摄像头 2", 2)
        self.source_combo.addItem("本地视频文件...", "file")
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        top.addWidget(self.source_combo)

        self.video_path_label = QLabel("")
        self.video_path_label.setStyleSheet("color:#666;")
        top.addWidget(self.video_path_label)

        self.loop_check = QCheckBox("循环播放")
        self.loop_check.setChecked(True)
        self.loop_check.setToolTip("取消勾选：视频播放结束后自动停止识别")
        top.addWidget(self.loop_check)

        top.addStretch(1)
        self.start_btn = QPushButton("开始识别")
        self.start_btn.setStyleSheet("font-size:15px; padding:6px 18px; background:#2a8; color:white;")
        self.start_btn.clicked.connect(self._on_start)
        top.addWidget(self.start_btn)

        self.stop_btn = QPushButton("停止")
        self.stop_btn.setStyleSheet(
            "font-size:15px; padding:6px 18px; background:#c0392b; color:white; font-weight:bold;"
        )
        self.stop_btn.clicked.connect(self._on_stop)
        top.addWidget(self.stop_btn)
        root.addLayout(top)

        # 中部：画面 + 右侧面板
        mid = QHBoxLayout()
        mid.setSpacing(12)

        self.video_label = QLabel("请选择输入源后点击「开始识别」")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(760, 520)
        self.video_label.setStyleSheet(
            "background:#101010; color:#888; font-size:16px; border:1px solid #333;"
        )
        mid.addWidget(self.video_label, 3)

        panel = QVBoxLayout()
        panel.setSpacing(8)

        panel.addWidget(QLabel("购物车明细"))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["商品", "单价(¥)", "数量", "小计(¥)"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for col in (1, 2, 3):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        panel.addWidget(self.table, 1)

        self.total_label = QLabel("合计：¥0.00")
        self.total_label.setStyleSheet(
            "font-size:30px; font-weight:bold; color:#2a8; padding:6px 4px;"
        )
        panel.addWidget(self.total_label)

        self.unknown_label = QLabel("")
        self.unknown_label.setWordWrap(True)
        self.unknown_label.setStyleSheet("color:#d22; font-size:14px; min-height:44px;")
        panel.addWidget(self.unknown_label)

        self.register_btn = QPushButton("＋ 注册未注册商品")
        self.register_btn.setStyleSheet(
            "font-size:14px; padding:8px; background:#2980b9; color:white;"
        )
        self.register_btn.setEnabled(False)
        self.register_btn.clicked.connect(self._on_register)
        panel.addWidget(self.register_btn)

        btn_row = QHBoxLayout()
        self.settle_btn = QPushButton("结  算")
        self.settle_btn.setStyleSheet(
            "font-size:18px; padding:10px; background:#e67e22; color:white; font-weight:bold;"
        )
        self.settle_btn.clicked.connect(self._on_settle)
        btn_row.addWidget(self.settle_btn, 1)

        self.reset_btn = QPushButton("重  置")
        self.reset_btn.setStyleSheet("font-size:16px; padding:10px;")
        self.reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self.reset_btn, 1)
        panel.addLayout(btn_row)

        panel_widget = QWidget()
        panel_widget.setLayout(panel)
        panel_widget.setMinimumWidth(360)
        mid.addWidget(panel_widget, 1)
        root.addLayout(mid, 1)

        self.statusBar().showMessage("就绪")

    # ------------------------------------------------------------ 交互
    def _on_source_changed(self) -> None:
        is_file = self.source_combo.currentData() == "file"
        self.loop_check.setEnabled(is_file)  # 摄像头无需循环控制
        if is_file:
            path, _ = QFileDialog.getOpenFileName(
                self, "选择视频文件", str(PROJECT_ROOT / "video"),
                "视频 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
            )
            if path:
                self._selected_video = path
                self.video_path_label.setText(Path(path).name)
            else:
                self.source_combo.setCurrentIndex(0)
                self.video_path_label.setText("")
        else:
            self.video_path_label.setText("")

    def _resolve_source(self):
        data = self.source_combo.currentData()
        if data == "file":
            return getattr(self, "_selected_video", None)
        return data

    def _on_start(self) -> None:
        if self.worker.isRunning():
            return
        source = self._resolve_source()
        if source is None:
            QMessageBox.warning(self, "提示", "请先选择视频文件。")
            return
        self.worker.video_loop = self.loop_check.isChecked()
        self.worker.set_source(source)
        self.worker.start()
        self._set_running_state(True)
        loop_note = "循环播放" if self.worker.video_loop else "播放完自动停止"
        self.statusBar().showMessage(f"识别中：{source}（{loop_note}）")

    def _on_stop(self) -> None:
        self.worker.request_stop()
        self.worker.wait(2000)
        self._set_running_state(False)
        self.statusBar().showMessage("已停止")

    def _on_reset(self) -> None:
        self.worker.reset_cart()
        self._last_cart = None
        self._refresh_table(None)
        self.statusBar().showMessage("购物车已重置")

    def _on_settle(self) -> None:
        cart = self._last_cart
        if not cart or cart["total_quantity"] == 0:
            QMessageBox.information(self, "结算", "购物车为空，无需结算。")
            return
        lines = [f"<b>共 {cart['total_quantity']} 件，合计 ¥{cart['total_amount']:.2f}</b><br/><br/>"]
        for item in cart["details"]:
            lines.append(
                f"{item['name']}　¥{item['unit_price']:.2f} × {item['quantity']} = "
                f"¥{item['amount']:.2f}<br/>"
            )
        QMessageBox.information(self, "结算单", "".join(lines))

    def _on_register(self) -> None:
        if self._last_unknown_group is None:
            return
        dialog = RegisterDialog(self._last_unknown_group, self.worker, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.worker.request_reload_library()
            self.statusBar().showMessage(
                f"注册成功：{dialog.result_model_class}（特征库已更新，识别下一帧生效）"
            )

    # ------------------------------------------------------------ 信号槽
    def on_frame_ready(self, annotated, detections, cart, fps) -> None:
        self._last_cart = cart
        self.video_label.setPixmap(self._to_pixmap(annotated))
        self._refresh_table(cart)

        unknown_items = [d for d in detections if not d["found"]]
        if unknown_items:
            last = unknown_items[-1]
            self._last_unknown_group = last["label"]
            group_name = GROUP_LABELS.get(last["label"], last["label"])
            self._last_unknown = f"{group_name}商品未注册（{last['unknown_reason']}）"
            self.unknown_label.setText(f"⚠ 未注册：{self._last_unknown}")
            self.register_btn.setEnabled(True)
        else:
            self.unknown_label.setText("")
            self.register_btn.setEnabled(False)

        self.statusBar().showMessage(f"识别中... FPS: {fps:.1f} | {len(detections)}个目标")

    def on_worker_error(self, message: str) -> None:
        self._set_running_state(False)
        QMessageBox.critical(self, "错误", message)

    def on_input_ended(self) -> None:
        self._set_running_state(False)
        self.statusBar().showMessage("视频播放结束，已停止识别（可重新开始）")

    # ------------------------------------------------------------ 工具
    def _refresh_table(self, cart) -> None:
        if not cart or not cart["details"]:
            self.table.setRowCount(0)
            self.total_label.setText("合计：¥0.00")
            return
        self.table.setRowCount(len(cart["details"]))
        for row, item in enumerate(cart["details"]):
            self.table.setItem(row, 0, QTableWidgetItem(item["name"]))
            self.table.setItem(row, 1, QTableWidgetItem(f"{item['unit_price']:.2f}"))
            self.table.setItem(row, 2, QTableWidgetItem(str(item["quantity"])))
            self.table.setItem(row, 3, QTableWidgetItem(f"{item['amount']:.2f}"))
        self.total_label.setText(f"合计：¥{cart['total_amount']:.2f}")

    def _to_pixmap(self, bgr: np.ndarray) -> QPixmap:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(image.copy())
        return pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _set_running_state(self, running: bool) -> None:
        self.start_btn.setEnabled(not running)
        self.stop_btn.setEnabled(running)
        self.source_combo.setEnabled(not running)
        self.loop_check.setEnabled(
            (not running) and self.source_combo.currentData() == "file"
        )

    def closeEvent(self, event) -> None:  # noqa: N802
        self.worker.request_stop()
        self.worker.wait(2000)
        event.accept()


class RegisterDialog(QDialog):
    """注册未注册商品：采集样本 + 填信息 → 入库 + 更新特征库。"""

    def __init__(self, group: str, worker: InferenceWorker, parent=None) -> None:
        super().__init__(parent)
        self.group = group
        self.worker = worker
        self.result_model_class: str = ""
        self.setWindowTitle("注册未注册商品")
        self.setMinimumWidth(420)

        form = QFormLayout(self)
        form.addRow("包装类型", QLabel(GROUP_LABELS.get(group, group)))

        # 样本采集提示
        self.sample_label = QLabel("样本：0 张")
        self.refresh_btn = QPushButton("刷新样本（多角度拍摄后点击）")
        self.refresh_btn.clicked.connect(self._refresh_sample_count)
        sample_row = QHBoxLayout()
        sample_row.addWidget(self.sample_label, 1)
        sample_row.addWidget(self.refresh_btn)
        form.addRow("采集样本", sample_row)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("商品名称，如：冰红茶")
        form.addRow("商品名称 *", self.name_edit)

        self.price_edit = QDoubleSpinBox()
        self.price_edit.setRange(0.0, 9999.0)
        self.price_edit.setDecimals(2)
        self.price_edit.setSingleStep(0.5)
        form.addRow("单价(¥) *", self.price_edit)

        dao = GoodsDao()
        suggested_sku = suggest_sku(dao, group)
        suggested_name = self.name_edit.text()
        suggested_model = suggest_model_class(group, suggested_sku, suggested_name)
        dao.close()

        self.sku_edit = QLineEdit(suggested_sku)
        form.addRow("SKU 编码", self.sku_edit)

        self.model_edit = QLineEdit(suggested_model)
        self.model_edit.setPlaceholderText("检索用分类名，建议自动生成")
        form.addRow("分类名", self.model_edit)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认注册")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self._refresh_sample_count()

    def _refresh_sample_count(self) -> None:
        count = len(self.worker.get_unknown_crops(self.group))
        self.sample_label.setText(
            f"样本：{count} 张（已自动采集该未注册商品的历史画面，建议 ≥5 张）"
        )

    def _on_accept(self) -> None:
        name = self.name_edit.text().strip()
        price = self.price_edit.value()
        sku = self.sku_edit.text().strip()
        model_class = self.model_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请填写商品名称。")
            return
        if not sku:
            QMessageBox.warning(self, "提示", "请填写 SKU 编码。")
            return
        crops = self.worker.get_unknown_crops(self.group)
        if not crops:
            QMessageBox.warning(self, "提示", "未采集到样本，请先把商品放到摄像头前识别为未注册后再注册。")
            return
        try:
            result = register_sku(
                group=self.group,
                sku=sku,
                product_name=name,
                unit_price=price,
                crops_bgr=crops,
                model_class=model_class or None,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "注册失败", str(exc))
            return
        self.result_model_class = result["model_class"]
        QMessageBox.information(
            self,
            "注册成功",
            f"已注册：{name}（¥{price:.2f}）\n"
            f"SKU：{result['sku']}\n分类名：{result['model_class']}\n"
            f"特征库新增样本：{result['samples_added']} 张\n\n"
            "识别下一帧起即可结算该商品。",
        )
        self.accept()


def main() -> int:
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
