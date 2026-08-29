"""不打开摄像头的 Qt 界面冒烟测试。"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QTimer  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from widget import Widget  # noqa: E402


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = Widget()
    window.show()
    required = (
        "sourceCombo",
        "startButton",
        "pauseButton",
        "stopButton",
        "videoLabel",
        "cartTable",
        "totalAmountLabel",
        "providerBadge",
        "registerButton",
    )
    missing = [name for name in required if window.findChild(object, name) is None]
    if missing:
        print("[FAIL] 缺少界面控件：", ", ".join(missing))
        return 1
    if window.worker.isRunning():
        print("[FAIL] 窗口初始化时不应自动启动推理")
        return 1
    if window.ui.unknownCard.isVisible() or window.register_button.isEnabled():
        print("[FAIL] 未检测到未知商品时注册入口不应启用")
        return 1
    print("[OK] Qt 主窗口创建成功")
    print("[OK] 摄像头和推理线程均未启动")
    print(f"[OK] 窗口标题：{window.windowTitle()}")
    app.processEvents()
    preview = Path(tempfile.gettempdir()) / "smart_checkout_qt_preview.png"
    if not window.grab().save(str(preview)):
        print("[FAIL] 无法生成界面预览图")
        return 1
    print(f"[OK] 界面预览：{preview}")
    QTimer.singleShot(100, app.quit)
    app.exec()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
