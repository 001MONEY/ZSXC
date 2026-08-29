"""未注册商品注册弹窗：录入商品信息并增量更新特征库。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
)

from database.goods_dao import GoodsDao
from feature_library_updater import (
    MIN_REGISTRATION_SAMPLES,
    collect_video_crops,
    register_sku,
    suggest_model_class,
    suggest_sku,
)
from inference_controller import PACKAGE_NAMES


DIALOG_STYLE = """
QDialog { background:#0d1929; color:#e5edf7; font-family:"Microsoft YaHei UI"; }
QLabel { color:#cbd8e6; }
QLabel#sampleHint { color:#8fa5bd; }
QLineEdit, QDoubleSpinBox {
    color:#eef5fc; background:#0a1523; border:1px solid #30465f;
    border-radius:7px; padding:7px 9px; min-height:22px;
}
QLineEdit:focus, QDoubleSpinBox:focus { border-color:#20c997; }
QPushButton {
    color:#dbe7f4; background:#1a2b40; border:1px solid #30465f;
    border-radius:7px; padding:7px 14px; font-weight:600;
}
QPushButton:hover { background:#233a54; border-color:#4b6684; }
QPushButton[text="确认注册"] { color:#06231c; background:#20c997; border-color:#20c997; }
"""


class RegisterProductDialog(QDialog):
    """用当前推理线程缓存的裁剪样本注册一个新 SKU。"""

    def __init__(
        self,
        package_type: str,
        worker: Any,
        parent=None,
        source_video: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self.package_type = package_type
        self.detected_package_type = package_type
        self.worker = worker
        self.result_model_class = ""
        self.result_sku = ""
        self._model_manually_edited = False
        self.video_path = Path(source_video) if source_video else None

        self.setWindowTitle("注册未注册商品")
        self.setMinimumWidth(500)
        self.setStyleSheet(DIALOG_STYLE)

        form = QFormLayout(self)
        form.setContentsMargins(24, 22, 24, 22)
        form.setHorizontalSpacing(16)
        form.setVerticalSpacing(13)
        self.package_combo = QComboBox()
        for key, display_name in PACKAGE_NAMES.items():
            self.package_combo.addItem(display_name, key)
        self.package_combo.setCurrentIndex(
            max(0, self.package_combo.findData(package_type))
        )
        form.addRow("实际包装类型", self.package_combo)

        self.sample_label = QLabel()
        self.sample_label.setObjectName("sampleHint")
        refresh_button = QPushButton("刷新样本")
        refresh_button.clicked.connect(self._refresh_sample_count)
        sample_row = QHBoxLayout()
        sample_row.addWidget(self.sample_label, 1)
        sample_row.addWidget(refresh_button)
        form.addRow("采集样本", sample_row)

        self.video_label = QLabel()
        self.video_label.setObjectName("sampleHint")
        self.video_label.setWordWrap(True)
        video_button = QPushButton("选择/更换视频")
        video_button.clicked.connect(self._choose_supplement_video)
        video_row = QHBoxLayout()
        video_row.addWidget(self.video_label, 1)
        video_row.addWidget(video_button)
        form.addRow("多姿态视频", video_row)

        self.name_edit = QLineEdit()
        self.name_edit.setPlaceholderText("例如：阿萨姆奶茶")
        form.addRow("商品名称 *", self.name_edit)

        self.price_edit = QDoubleSpinBox()
        self.price_edit.setRange(0.01, 9999.00)
        self.price_edit.setDecimals(2)
        self.price_edit.setSingleStep(0.50)
        self.price_edit.setPrefix("¥ ")
        form.addRow("商品单价 *", self.price_edit)

        dao = GoodsDao()
        try:
            suggested_sku = suggest_sku(dao, package_type)
        finally:
            dao.close()
        self.sku_edit = QLineEdit(suggested_sku)
        form.addRow("SKU 编码 *", self.sku_edit)

        self.model_edit = QLineEdit(
            suggest_model_class(package_type, suggested_sku, "")
        )
        self.model_edit.setPlaceholderText("特征检索类别名")
        form.addRow("检索分类名 *", self.model_edit)

        note = QLabel(
            "系统会跨时间筛选未知画面，并对补充视频均匀抽帧、过滤模糊和重复样本。"
            "建议视频只拍一件商品，并覆盖正面、侧面、远近和横竖姿态。"
        )
        note.setWordWrap(True)
        note.setObjectName("sampleHint")
        form.addRow("说明", note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("确认注册")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self._register)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.name_edit.textChanged.connect(self._refresh_model_suggestion)
        self.sku_edit.textChanged.connect(self._refresh_model_suggestion)
        self.model_edit.textEdited.connect(self._mark_model_edited)
        self.package_combo.currentIndexChanged.connect(self._package_changed)
        self._refresh_sample_count()
        self._refresh_video_label()

    def _mark_model_edited(self, _text: str) -> None:
        self._model_manually_edited = True

    def _package_changed(self, _index: int) -> None:
        package_type = self.package_combo.currentData()
        if not package_type or package_type == self.package_type:
            return
        self.package_type = package_type
        dao = GoodsDao()
        try:
            suggested_sku = suggest_sku(dao, package_type)
        finally:
            dao.close()
        self._model_manually_edited = False
        self.sku_edit.setText(suggested_sku)
        self._refresh_model_suggestion()

    def _refresh_model_suggestion(self, _text: str = "") -> None:
        if self._model_manually_edited:
            return
        self.model_edit.setText(
            suggest_model_class(
                self.package_type,
                self.sku_edit.text().strip(),
                self.name_edit.text().strip(),
            )
        )

    def _refresh_sample_count(self) -> None:
        count = len(self.worker.get_unknown_crops(self.detected_package_type))
        state = "可注册" if count >= MIN_REGISTRATION_SAMPLES else "建议继续多角度采集"
        self.sample_label.setText(f"{count} 张时间分散样本 · {state}")

    def _refresh_video_label(self) -> None:
        if self.video_path is None:
            self.video_label.setText("未选择（将仅使用实时缓存）")
            self.video_label.setToolTip("")
        else:
            self.video_label.setText(self.video_path.name)
            self.video_label.setToolTip(str(self.video_path))

    def _choose_supplement_video(self) -> None:
        initial = str(self.video_path.parent if self.video_path else Path.cwd() / "video")
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择新商品多姿态视频",
            initial,
            "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)",
        )
        if path:
            self.video_path = Path(path)
            self._refresh_video_label()

    def _register(self) -> None:
        name = self.name_edit.text().strip()
        sku = self.sku_edit.text().strip()
        model_class = self.model_edit.text().strip()
        live_crops = self.worker.get_unknown_crops(self.detected_package_type)
        if not name:
            QMessageBox.warning(self, "信息不完整", "请填写商品名称。")
            return
        if not sku or not model_class:
            QMessageBox.warning(self, "信息不完整", "SKU 编码和检索分类名不能为空。")
            return
        if not live_crops and self.video_path is None:
            QMessageBox.warning(
                self,
                "没有样本",
                "尚未采集到该商品的裁剪图，请先让它在识别画面中保持几秒。",
            )
            return

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            crops = list(live_crops)
            video_stats = None
            if self.video_path is not None:
                self.sample_label.setText("正在从视频均匀提取多姿态样本…")
                QApplication.processEvents()
                video_crops, video_stats = collect_video_crops(
                    self.video_path,
                    self.package_type,
                )
                crops.extend(video_crops)
            result = register_sku(
                group=self.package_type,
                sku=sku,
                product_name=name,
                unit_price=self.price_edit.value(),
                crops_bgr=crops,
                model_class=model_class,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "注册失败", str(exc))
            return
        finally:
            QApplication.restoreOverrideCursor()

        self.result_model_class = result["model_class"]
        self.result_sku = result["sku"]
        QMessageBox.information(
            self,
            "注册成功",
            f"{name}（¥{self.price_edit.value():.2f}）已加入商品库。\n"
            f"SKU：{result['sku']}\n"
            f"检索分类名：{result['model_class']}\n"
            f"候选样本：{result['received']} 张\n"
            f"筛选入库：{result['samples_added']} 张\n"
            f"多姿态原型：{result['target_prototypes']} 个\n"
            + (
                f"视频检测：{video_stats['detected_frames']}/"
                f"{video_stats['sampled_frames']} 个采样帧\n"
                if video_stats is not None
                else ""
            )
            + "\n"
            "特征库将自动热重载，无需重启程序。",
        )
        self.accept()
