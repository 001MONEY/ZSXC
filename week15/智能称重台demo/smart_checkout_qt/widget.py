"""SmartCheckout Qt 主窗口。"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QCloseEvent, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHeaderView,
    QMessageBox,
    QPushButton,
    QTableWidgetItem,
    QWidget,
)

from inference_controller import InferenceController, PACKAGE_NAMES, PROJECT_ROOT
from registration_dialog import RegisterProductDialog
from ui_form import Ui_Widget


APP_STYLE = """
QWidget#Widget {
    background: #08111f;
    color: #e5edf7;
    font-family: "Microsoft YaHei UI";
    font-size: 14px;
}
QFrame#headerCard, QFrame#controlCard, QFrame#videoCard,
QFrame#recognitionCard, QFrame#cartCard, QFrame#totalCard {
    background: #101d2e;
    border: 1px solid #22344b;
    border-radius: 12px;
}
QFrame#headerCard {
    background: #0d2034;
    border-left: 4px solid #20c997;
}
QLabel#titleLabel { color: #f8fbff; font-size: 25px; font-weight: 700; }
QLabel#subtitleLabel { color: #7f94ad; font-size: 12px; }
QLabel#providerBadge {
    color: #9fb2c8; background: #17283d; border: 1px solid #2a405a;
    border-radius: 15px; padding: 7px 15px; font-weight: 600;
}
QLabel#sourceCaption, QLabel#recognitionCaption, QLabel#thresholdCaption,
QLabel#totalCaption { color: #92a6bd; }
QLabel#sourcePathLabel {
    color: #8da2ba; background: #0b1726; border-radius: 7px; padding: 0 12px;
}
QComboBox {
    color: #eef5fc; background: #0b1726; border: 1px solid #30465f;
    border-radius: 7px; padding: 6px 12px;
}
QComboBox:hover, QComboBox:focus { border-color: #20c997; }
QComboBox QAbstractItemView {
    color: #e5edf7; background: #101d2e; selection-background-color: #1d6f62;
    border: 1px solid #30465f; outline: 0;
}
QPushButton {
    color: #dbe7f4; background: #1a2b40; border: 1px solid #30465f;
    border-radius: 8px; padding: 6px 14px; font-weight: 600;
}
QPushButton:hover { background: #233a54; border-color: #4b6684; }
QPushButton:pressed { background: #102033; }
QPushButton:disabled { color: #596b80; background: #111e2d; border-color: #203044; }
QPushButton#startButton { color: #06231c; background: #20c997; border-color: #20c997; }
QPushButton#startButton:hover { background: #3dd9aa; }
QPushButton#stopButton { color: #fecaca; border-color: #7f3540; }
QPushButton#registerButton {
    color:#ffedd5; background:#71351f; border-color:#a75430; padding:6px 10px;
}
QPushButton#registerButton:hover { background:#8a4227; border-color:#d97745; }
QPushButton#settleButton { color: #061b17; background: #34d399; border-color: #34d399; font-size: 17px; }
QPushButton#settleButton:hover { background: #55e0b1; }
QLabel#videoTitleLabel, QLabel#cartTitleLabel { color: #f0f6fc; font-size: 16px; font-weight: 700; }
QLabel#liveDot { color: #52667d; font-size: 18px; }
QLabel#detectionBadge, QLabel#fpsBadge, QLabel#cartCountBadge {
    color: #a9bdd3; background: #17283d; border-radius: 11px; padding: 4px 10px;
}
QFrame#videoViewport { background: #03070d; border: 1px solid #1e3045; border-radius: 9px; }
QLabel#videoLabel { color: #657b93; font-size: 16px; padding: 24px; }
QLabel#recognitionState { color: #34d399; font-size: 15px; font-weight: 700; }
QLabel#thresholdLabel { color: #a9bdd3; font-family: Consolas, monospace; font-size: 12px; }
QTableWidget {
    color: #dce7f3; background: #0c1828; alternate-background-color: #101f31;
    border: 0; border-radius: 7px; gridline-color: #21344a;
}
QTableWidget::item { padding: 9px 5px; border-bottom: 1px solid #1b2b3f; }
QHeaderView::section {
    color: #8fa5bd; background: #14253a; border: 0; border-bottom: 1px solid #2b4058;
    padding: 9px 5px; font-weight: 600;
}
QFrame#unknownCard { background: #382319; border: 1px solid #8b4a2e; border-radius: 10px; }
QLabel#unknownIcon {
    color: #2b160c; background: #fb923c; min-width: 26px; max-width: 26px;
    min-height: 26px; max-height: 26px; border-radius: 13px; font-weight: 900;
}
QLabel#unknownLabel { color: #fed7aa; }
QFrame#totalCard { background: #0e2b29; border-color: #1d5e55; }
QLabel#totalQuantityLabel { color: #8aa9a5; font-size: 12px; }
QLabel#totalAmountLabel { color: #4adeb4; font-size: 34px; font-weight: 800; }
QFrame#footerBar { background: #0b1726; border-radius: 7px; }
QLabel#statusLabel, QLabel#modelLabel { color: #748aa3; font-size: 12px; }
QSplitter::handle { background: transparent; width: 10px; }
"""


class Widget(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.ui = Ui_Widget()
        self.ui.setupUi(self)
        self.setStyleSheet(APP_STYLE)

        self._selected_video: Path | None = None
        self._last_frame: np.ndarray | None = None
        self._last_cart: dict | None = None
        self._running = False
        self._paused = False
        self._last_unknown_group: str | None = None

        self.worker = InferenceController(self)
        self._configure_widgets()
        self._connect_signals()
        self._set_running(False)
        self.ui.unknownCard.hide()

    def _configure_widgets(self) -> None:
        self.ui.contentSplitter.setSizes([940, 420])
        for badge in (
            self.ui.liveDot,
            self.ui.videoTitleLabel,
            self.ui.detectionBadge,
            self.ui.fpsBadge,
        ):
            badge.setFixedHeight(30)
        self.ui.detectionBadge.setMinimumWidth(72)
        self.ui.fpsBadge.setMinimumWidth(94)
        table = self.ui.cartTable
        table.verticalHeader().hide()
        table.verticalHeader().setDefaultSectionSize(46)
        table.setEditTriggers(table.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3):
            table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.ResizeToContents
            )
        self.ui.chooseVideoButton.setEnabled(False)
        self.register_button = QPushButton("＋ 注册商品", self.ui.unknownCard)
        self.register_button.setObjectName("registerButton")
        self.register_button.setMinimumWidth(108)
        self.register_button.setToolTip("将当前未注册商品加入数据库和特征库")
        self.register_button.setEnabled(False)
        self.ui.unknownLayout.addWidget(self.register_button)

    def _connect_signals(self) -> None:
        self.ui.sourceCombo.currentIndexChanged.connect(self._source_changed)
        self.ui.chooseVideoButton.clicked.connect(self._choose_video)
        self.ui.startButton.clicked.connect(self._start)
        self.ui.pauseButton.clicked.connect(self._toggle_pause)
        self.ui.stopButton.clicked.connect(self._stop)
        self.ui.resetButton.clicked.connect(self._reset_cart)
        self.ui.settleButton.clicked.connect(self._settle)
        self.register_button.clicked.connect(self._register_unknown_product)

        self.worker.frame_ready.connect(self._show_frame)
        self.worker.state_changed.connect(self._show_worker_state)
        self.worker.provider_ready.connect(self._show_provider)
        self.worker.catalog_ready.connect(self._show_catalog_size)
        self.worker.failed.connect(self._show_error)
        self.worker.source_finished.connect(self._source_finished)
        self.worker.finished.connect(self._worker_finished)

    def _source_changed(self, index: int) -> None:
        is_file = index == 3
        self.ui.chooseVideoButton.setEnabled(is_file and not self._running)
        if is_file:
            if self._selected_video is None:
                self.ui.sourcePathLabel.setText("尚未选择视频")
            else:
                self.ui.sourcePathLabel.setText(self._selected_video.name)
                self.ui.sourcePathLabel.setToolTip(str(self._selected_video))
        else:
            self.ui.sourcePathLabel.setText(f"当前：摄像头 {index}")
            self.ui.sourcePathLabel.setToolTip("")

    def _choose_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择测试视频",
            str(PROJECT_ROOT / "video"),
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if not path:
            return
        self._selected_video = Path(path)
        self.ui.sourceCombo.setCurrentIndex(3)
        self.ui.sourcePathLabel.setText(self._selected_video.name)
        self.ui.sourcePathLabel.setToolTip(str(self._selected_video))
        self.ui.statusLabel.setText(f"已选择视频：{self._selected_video}")

    def _resolve_source(self) -> int | str | None:
        index = self.ui.sourceCombo.currentIndex()
        if index == 3:
            return str(self._selected_video) if self._selected_video else None
        return index

    def _start(self) -> None:
        if self.worker.isRunning():
            return
        source = self._resolve_source()
        if source is None:
            QMessageBox.warning(self, "未选择视频", "请先点击“选择视频”选择本地文件。")
            return
        self.worker.set_source(source)
        self._paused = False
        self.ui.pauseButton.setText("Ⅱ  暂停")
        self._set_running(True)
        self.ui.recognitionState.setText("启动中")
        self.ui.statusLabel.setText("正在启动后台推理线程…")
        self.worker.start()

    def _toggle_pause(self) -> None:
        if not self.worker.isRunning():
            return
        self._paused = not self._paused
        self.worker.set_paused(self._paused)
        if self._paused:
            self.ui.pauseButton.setText("▶  继续")
            self.ui.recognitionState.setText("已暂停")
            self.ui.statusLabel.setText("识别已暂停，当前购物车状态已保留")
            self.ui.liveDot.setStyleSheet("color: #fbbf24;")
        else:
            self.ui.pauseButton.setText("Ⅱ  暂停")
            self.ui.recognitionState.setText("识别运行中")
            self.ui.statusLabel.setText("已继续识别")
            self.ui.liveDot.setStyleSheet("color: #34d399;")

    def _stop(self) -> None:
        if not self.worker.isRunning():
            return
        self.ui.statusLabel.setText("正在停止输入源…")
        self.ui.stopButton.setEnabled(False)
        self.worker.request_stop()

    def _set_running(self, running: bool) -> None:
        self._running = running
        self.ui.startButton.setEnabled(not running)
        self.ui.pauseButton.setEnabled(running)
        self.ui.stopButton.setEnabled(running)
        self.ui.sourceCombo.setEnabled(not running)
        self.ui.chooseVideoButton.setEnabled(not running and self.ui.sourceCombo.currentIndex() == 3)
        self.ui.liveDot.setStyleSheet("color: #34d399;" if running else "color: #52667d;")

    def _show_worker_state(self, state: str) -> None:
        self.ui.recognitionState.setText(state)
        self.ui.statusLabel.setText(state)

    def _show_provider(self, provider: str) -> None:
        self.ui.providerBadge.setText(f"●  ONNX · {provider}")
        if provider == "CUDA GPU":
            self.ui.providerBadge.setStyleSheet(
                "color:#86efcf; background:#123831; border:1px solid #236b5e; "
                "border-radius:15px; padding:7px 15px; font-weight:600;"
            )
        else:
            self.ui.providerBadge.setStyleSheet(
                "color:#fde68a; background:#3a2d16; border:1px solid #73591f; "
                "border-radius:15px; padding:7px 15px; font-weight:600;"
            )

    def _show_catalog_size(self, count: int) -> None:
        self.ui.modelLabel.setText(f"ONNX · 4包装类别 · {count} SKU")

    def _show_frame(
        self,
        frame: np.ndarray,
        detections: list[dict],
        cart: dict,
        fps: float,
    ) -> None:
        self._last_frame = frame.copy()
        self._last_cart = cart
        self._render_frame()
        self._render_cart(cart)

        unknowns = [item for item in detections if not item["found"]]
        self.ui.detectionBadge.setText(f"目标 {len(detections)}")
        self.ui.fpsBadge.setText(f"{fps:.1f} FPS")
        if unknowns:
            current_unknown = unknowns[-1]
            self._last_unknown_group = current_unknown["package_type"]
            package_types = sorted({PACKAGE_NAMES[item["package_type"]] for item in unknowns})
            reasons = sorted({item["reason"] for item in unknowns})
            self.ui.unknownLabel.setText(
                f"检测到未注册的{'/'.join(package_types)}商品，不计入购物车\n"
                f"判定依据：{'；'.join(reasons)}"
            )
            self.register_button.setText(
                f"＋ 注册{PACKAGE_NAMES[self._last_unknown_group]}商品"
            )
            self.register_button.setEnabled(True)
            self.ui.unknownCard.show()
        else:
            self._last_unknown_group = None
            self.register_button.setEnabled(False)
            self.ui.unknownCard.hide()

    def _register_unknown_product(self) -> None:
        if self._last_unknown_group is None:
            return
        was_paused = self._paused
        if self.worker.isRunning() and not was_paused:
            self.worker.set_paused(True)
        try:
            source_video = (
                self._selected_video
                if self.ui.sourceCombo.currentIndex() == 3
                else None
            )
            dialog = RegisterProductDialog(
                self._last_unknown_group,
                self.worker,
                self,
                source_video=source_video,
            )
        except Exception as exc:  # noqa: BLE001
            if self.worker.isRunning() and not was_paused:
                self.worker.set_paused(False)
            QMessageBox.critical(self, "无法打开注册界面", str(exc))
            return
        if dialog.exec() != dialog.DialogCode.Accepted:
            if self.worker.isRunning() and not was_paused:
                self.worker.set_paused(False)
            return
        self.worker.clear_unknown_crops(self._last_unknown_group)
        self.worker.reset_cart()
        self.worker.request_reload_library()
        if self.worker.isRunning() and not was_paused:
            self.worker.set_paused(False)
        self.register_button.setEnabled(False)
        self.ui.statusLabel.setText(
            f"注册成功：{dialog.result_model_class}，正在热重载特征库…"
        )

    def _render_frame(self) -> None:
        if self._last_frame is None:
            return
        rgb = cv2.cvtColor(self._last_frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        image = QImage(
            rgb.data,
            width,
            height,
            channels * width,
            QImage.Format.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.ui.videoLabel.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.ui.videoLabel.setPixmap(pixmap)

    def _render_cart(self, cart: dict | None) -> None:
        details = cart.get("details", []) if cart else []
        self.ui.cartTable.setRowCount(len(details))
        for row, item in enumerate(details):
            values = (
                item["name"],
                f"¥{item['unit_price']:.2f}",
                str(item["quantity"]),
                f"¥{item['amount']:.2f}",
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if column > 0:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.ui.cartTable.setItem(row, column, cell)

        quantity = int(cart.get("total_quantity", 0)) if cart else 0
        amount = float(cart.get("total_amount", 0.0)) if cart else 0.0
        self.ui.cartCountBadge.setText(f"{quantity} 件")
        self.ui.totalQuantityLabel.setText(f"共 {quantity} 件商品")
        self.ui.totalAmountLabel.setText(f"¥{amount:.2f}")

    def _reset_cart(self) -> None:
        self.worker.reset_cart()
        self._last_cart = None
        self._render_cart(None)
        self.ui.unknownCard.hide()
        self._last_unknown_group = None
        self.register_button.setEnabled(False)
        self.ui.statusLabel.setText("购物车已重置")

    def _settle(self) -> None:
        cart = self._last_cart
        if not cart or not cart.get("details"):
            QMessageBox.information(self, "结算", "当前购物车为空。")
            return
        rows = "".join(
            f"<tr><td>{item['name']}</td><td>¥{item['unit_price']:.2f}</td>"
            f"<td>× {item['quantity']}</td><td><b>¥{item['amount']:.2f}</b></td></tr>"
            for item in cart["details"]
        )
        html = (
            "<h2>结算确认</h2><table cellspacing='8'>"
            f"{rows}</table><hr>"
            f"<p>共 {cart['total_quantity']} 件，"
            f"<span style='font-size:22px;color:#149d78'><b>合计 ¥{cart['total_amount']:.2f}</b></span></p>"
        )
        QMessageBox.information(self, "智能称重台 · 结算", html)

    def _show_error(self, message: str) -> None:
        self.ui.statusLabel.setText(f"错误：{message}")
        self.ui.recognitionState.setText("运行错误")
        QMessageBox.critical(self, "运行错误", message)

    def _source_finished(self, reason: str) -> None:
        self.ui.statusLabel.setText(reason)
        self.ui.recognitionState.setText(reason)

    def _worker_finished(self) -> None:
        self._paused = False
        self.ui.pauseButton.setText("Ⅱ  暂停")
        self._set_running(False)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._render_frame()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        if self.worker.isRunning():
            self.worker.request_stop()
            if not self.worker.wait(5000):
                QMessageBox.warning(self, "正在关闭", "推理线程仍在释放资源，请稍候再关闭。")
                event.ignore()
                return
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("SmartCheckout")
    app.setOrganizationName("SmartCheckout Team")
    app.setStyle("Fusion")
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = Widget()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
