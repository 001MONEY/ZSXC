# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'form.ui'
##
## Created by: Qt User Interface Compiler version 6.11.2
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QComboBox, QFrame,
    QGridLayout, QHBoxLayout, QHeaderView, QLabel,
    QPushButton, QSizePolicy, QSpacerItem, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

class Ui_Widget(object):
    def setupUi(self, Widget):
        if not Widget.objectName():
            Widget.setObjectName(u"Widget")
        Widget.resize(1440, 900)
        Widget.setMinimumSize(QSize(1180, 720))
        self.rootLayout = QVBoxLayout(Widget)
        self.rootLayout.setSpacing(14)
        self.rootLayout.setObjectName(u"rootLayout")
        self.rootLayout.setContentsMargins(20, 18, 20, 14)
        self.headerCard = QFrame(Widget)
        self.headerCard.setObjectName(u"headerCard")
        self.headerCard.setMinimumSize(QSize(0, 78))
        self.headerCard.setMaximumSize(QSize(16777215, 78))
        self.headerLayout = QHBoxLayout(self.headerCard)
        self.headerLayout.setObjectName(u"headerLayout")
        self.headerLayout.setContentsMargins(22, -1, 22, -1)
        self.titleLayout = QVBoxLayout()
        self.titleLayout.setSpacing(2)
        self.titleLayout.setObjectName(u"titleLayout")
        self.titleLabel = QLabel(self.headerCard)
        self.titleLabel.setObjectName(u"titleLabel")

        self.titleLayout.addWidget(self.titleLabel)

        self.subtitleLabel = QLabel(self.headerCard)
        self.subtitleLabel.setObjectName(u"subtitleLabel")

        self.titleLayout.addWidget(self.subtitleLabel)


        self.headerLayout.addLayout(self.titleLayout)

        self.headerSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.headerLayout.addItem(self.headerSpacer)

        self.providerBadge = QLabel(self.headerCard)
        self.providerBadge.setObjectName(u"providerBadge")
        self.providerBadge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.headerLayout.addWidget(self.providerBadge)


        self.rootLayout.addWidget(self.headerCard)

        self.controlCard = QFrame(Widget)
        self.controlCard.setObjectName(u"controlCard")
        self.controlCard.setMinimumSize(QSize(0, 76))
        self.controlCard.setMaximumSize(QSize(16777215, 76))
        self.controlLayout = QHBoxLayout(self.controlCard)
        self.controlLayout.setSpacing(10)
        self.controlLayout.setObjectName(u"controlLayout")
        self.controlLayout.setContentsMargins(16, -1, 16, -1)
        self.sourceCaption = QLabel(self.controlCard)
        self.sourceCaption.setObjectName(u"sourceCaption")

        self.controlLayout.addWidget(self.sourceCaption)

        self.sourceCombo = QComboBox(self.controlCard)
        self.sourceCombo.addItem("")
        self.sourceCombo.addItem("")
        self.sourceCombo.addItem("")
        self.sourceCombo.addItem("")
        self.sourceCombo.setObjectName(u"sourceCombo")
        self.sourceCombo.setMinimumSize(QSize(148, 40))

        self.controlLayout.addWidget(self.sourceCombo)

        self.chooseVideoButton = QPushButton(self.controlCard)
        self.chooseVideoButton.setObjectName(u"chooseVideoButton")
        self.chooseVideoButton.setMinimumSize(QSize(112, 40))

        self.controlLayout.addWidget(self.chooseVideoButton)

        self.sourcePathLabel = QLabel(self.controlCard)
        self.sourcePathLabel.setObjectName(u"sourcePathLabel")
        self.sourcePathLabel.setMinimumSize(QSize(220, 40))
        self.sourcePathLabel.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.controlLayout.addWidget(self.sourcePathLabel)

        self.controlSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.controlLayout.addItem(self.controlSpacer)

        self.startButton = QPushButton(self.controlCard)
        self.startButton.setObjectName(u"startButton")
        self.startButton.setMinimumSize(QSize(118, 42))

        self.controlLayout.addWidget(self.startButton)

        self.pauseButton = QPushButton(self.controlCard)
        self.pauseButton.setObjectName(u"pauseButton")
        self.pauseButton.setMinimumSize(QSize(96, 42))

        self.controlLayout.addWidget(self.pauseButton)

        self.stopButton = QPushButton(self.controlCard)
        self.stopButton.setObjectName(u"stopButton")
        self.stopButton.setMinimumSize(QSize(96, 42))

        self.controlLayout.addWidget(self.stopButton)


        self.rootLayout.addWidget(self.controlCard)

        self.contentSplitter = QSplitter(Widget)
        self.contentSplitter.setObjectName(u"contentSplitter")
        self.contentSplitter.setOrientation(Qt.Orientation.Horizontal)
        self.contentSplitter.setChildrenCollapsible(False)
        self.videoCard = QFrame(self.contentSplitter)
        self.videoCard.setObjectName(u"videoCard")
        self.videoCard.setMinimumSize(QSize(700, 0))
        self.videoCardLayout = QVBoxLayout(self.videoCard)
        self.videoCardLayout.setSpacing(10)
        self.videoCardLayout.setObjectName(u"videoCardLayout")
        self.videoCardLayout.setContentsMargins(14, 14, 14, 14)
        self.videoHeaderLayout = QHBoxLayout()
        self.videoHeaderLayout.setObjectName(u"videoHeaderLayout")
        self.liveDot = QLabel(self.videoCard)
        self.liveDot.setObjectName(u"liveDot")

        self.videoHeaderLayout.addWidget(self.liveDot)

        self.videoTitleLabel = QLabel(self.videoCard)
        self.videoTitleLabel.setObjectName(u"videoTitleLabel")

        self.videoHeaderLayout.addWidget(self.videoTitleLabel)

        self.videoHeaderSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.videoHeaderLayout.addItem(self.videoHeaderSpacer)

        self.detectionBadge = QLabel(self.videoCard)
        self.detectionBadge.setObjectName(u"detectionBadge")
        self.detectionBadge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoHeaderLayout.addWidget(self.detectionBadge)

        self.fpsBadge = QLabel(self.videoCard)
        self.fpsBadge.setObjectName(u"fpsBadge")
        self.fpsBadge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.videoHeaderLayout.addWidget(self.fpsBadge)


        self.videoCardLayout.addLayout(self.videoHeaderLayout)

        self.videoViewport = QFrame(self.videoCard)
        self.videoViewport.setObjectName(u"videoViewport")
        self.viewportLayout = QVBoxLayout(self.videoViewport)
        self.viewportLayout.setObjectName(u"viewportLayout")
        self.viewportLayout.setContentsMargins(0, 0, 0, 0)
        self.videoLabel = QLabel(self.videoViewport)
        self.videoLabel.setObjectName(u"videoLabel")
        self.videoLabel.setMinimumSize(QSize(640, 480))
        self.videoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.videoLabel.setWordWrap(True)

        self.viewportLayout.addWidget(self.videoLabel)


        self.videoCardLayout.addWidget(self.videoViewport)

        self.contentSplitter.addWidget(self.videoCard)
        self.rightPanel = QWidget(self.contentSplitter)
        self.rightPanel.setObjectName(u"rightPanel")
        self.rightPanel.setMinimumSize(QSize(390, 0))
        self.rightPanel.setMaximumSize(QSize(480, 16777215))
        self.rightPanelLayout = QVBoxLayout(self.rightPanel)
        self.rightPanelLayout.setSpacing(12)
        self.rightPanelLayout.setObjectName(u"rightPanelLayout")
        self.rightPanelLayout.setContentsMargins(4, 0, 0, 0)
        self.recognitionCard = QFrame(self.rightPanel)
        self.recognitionCard.setObjectName(u"recognitionCard")
        self.recognitionCard.setMinimumSize(QSize(0, 92))
        self.recognitionLayout = QGridLayout(self.recognitionCard)
        self.recognitionLayout.setObjectName(u"recognitionLayout")
        self.recognitionLayout.setContentsMargins(16, 13, 16, 13)
        self.recognitionCaption = QLabel(self.recognitionCard)
        self.recognitionCaption.setObjectName(u"recognitionCaption")

        self.recognitionLayout.addWidget(self.recognitionCaption, 0, 0, 1, 1)

        self.recognitionState = QLabel(self.recognitionCard)
        self.recognitionState.setObjectName(u"recognitionState")
        self.recognitionState.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.recognitionLayout.addWidget(self.recognitionState, 0, 1, 1, 1)

        self.thresholdCaption = QLabel(self.recognitionCard)
        self.thresholdCaption.setObjectName(u"thresholdCaption")

        self.recognitionLayout.addWidget(self.thresholdCaption, 1, 0, 1, 1)

        self.thresholdLabel = QLabel(self.recognitionCard)
        self.thresholdLabel.setObjectName(u"thresholdLabel")
        self.thresholdLabel.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.recognitionLayout.addWidget(self.thresholdLabel, 1, 1, 1, 1)


        self.rightPanelLayout.addWidget(self.recognitionCard)

        self.cartCard = QFrame(self.rightPanel)
        self.cartCard.setObjectName(u"cartCard")
        self.cartLayout = QVBoxLayout(self.cartCard)
        self.cartLayout.setSpacing(10)
        self.cartLayout.setObjectName(u"cartLayout")
        self.cartLayout.setContentsMargins(14, 14, 14, 14)
        self.cartHeaderLayout = QHBoxLayout()
        self.cartHeaderLayout.setObjectName(u"cartHeaderLayout")
        self.cartTitleLabel = QLabel(self.cartCard)
        self.cartTitleLabel.setObjectName(u"cartTitleLabel")

        self.cartHeaderLayout.addWidget(self.cartTitleLabel)

        self.cartHeaderSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.cartHeaderLayout.addItem(self.cartHeaderSpacer)

        self.cartCountBadge = QLabel(self.cartCard)
        self.cartCountBadge.setObjectName(u"cartCountBadge")
        self.cartCountBadge.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.cartHeaderLayout.addWidget(self.cartCountBadge)


        self.cartLayout.addLayout(self.cartHeaderLayout)

        self.cartTable = QTableWidget(self.cartCard)
        if (self.cartTable.columnCount() < 4):
            self.cartTable.setColumnCount(4)
        __qtablewidgetitem = QTableWidgetItem()
        self.cartTable.setHorizontalHeaderItem(0, __qtablewidgetitem)
        __qtablewidgetitem1 = QTableWidgetItem()
        self.cartTable.setHorizontalHeaderItem(1, __qtablewidgetitem1)
        __qtablewidgetitem2 = QTableWidgetItem()
        self.cartTable.setHorizontalHeaderItem(2, __qtablewidgetitem2)
        __qtablewidgetitem3 = QTableWidgetItem()
        self.cartTable.setHorizontalHeaderItem(3, __qtablewidgetitem3)
        self.cartTable.setObjectName(u"cartTable")
        self.cartTable.setAlternatingRowColors(False)
        self.cartTable.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.cartTable.setShowGrid(False)
        self.cartTable.setColumnCount(4)
        self.cartTable.setRowCount(0)

        self.cartLayout.addWidget(self.cartTable)


        self.rightPanelLayout.addWidget(self.cartCard)

        self.unknownCard = QFrame(self.rightPanel)
        self.unknownCard.setObjectName(u"unknownCard")
        self.unknownCard.setMinimumSize(QSize(0, 66))
        self.unknownLayout = QHBoxLayout(self.unknownCard)
        self.unknownLayout.setObjectName(u"unknownLayout")
        self.unknownLayout.setContentsMargins(14, -1, 14, -1)
        self.unknownIcon = QLabel(self.unknownCard)
        self.unknownIcon.setObjectName(u"unknownIcon")
        self.unknownIcon.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.unknownLayout.addWidget(self.unknownIcon)

        self.unknownLabel = QLabel(self.unknownCard)
        self.unknownLabel.setObjectName(u"unknownLabel")
        self.unknownLabel.setWordWrap(True)

        self.unknownLayout.addWidget(self.unknownLabel)


        self.rightPanelLayout.addWidget(self.unknownCard)

        self.totalCard = QFrame(self.rightPanel)
        self.totalCard.setObjectName(u"totalCard")
        self.totalLayout = QHBoxLayout(self.totalCard)
        self.totalLayout.setObjectName(u"totalLayout")
        self.totalLayout.setContentsMargins(18, 15, 18, 15)
        self.totalCaptionLayout = QVBoxLayout()
        self.totalCaptionLayout.setObjectName(u"totalCaptionLayout")
        self.totalCaption = QLabel(self.totalCard)
        self.totalCaption.setObjectName(u"totalCaption")

        self.totalCaptionLayout.addWidget(self.totalCaption)

        self.totalQuantityLabel = QLabel(self.totalCard)
        self.totalQuantityLabel.setObjectName(u"totalQuantityLabel")

        self.totalCaptionLayout.addWidget(self.totalQuantityLabel)


        self.totalLayout.addLayout(self.totalCaptionLayout)

        self.totalSpacer = QSpacerItem(30, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.totalLayout.addItem(self.totalSpacer)

        self.totalAmountLabel = QLabel(self.totalCard)
        self.totalAmountLabel.setObjectName(u"totalAmountLabel")
        self.totalAmountLabel.setAlignment(Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTrailing|Qt.AlignmentFlag.AlignVCenter)

        self.totalLayout.addWidget(self.totalAmountLabel)


        self.rightPanelLayout.addWidget(self.totalCard)

        self.checkoutLayout = QHBoxLayout()
        self.checkoutLayout.setSpacing(10)
        self.checkoutLayout.setObjectName(u"checkoutLayout")
        self.resetButton = QPushButton(self.rightPanel)
        self.resetButton.setObjectName(u"resetButton")
        self.resetButton.setMinimumSize(QSize(0, 50))

        self.checkoutLayout.addWidget(self.resetButton)

        self.settleButton = QPushButton(self.rightPanel)
        self.settleButton.setObjectName(u"settleButton")
        self.settleButton.setMinimumSize(QSize(0, 50))

        self.checkoutLayout.addWidget(self.settleButton)


        self.rightPanelLayout.addLayout(self.checkoutLayout)

        self.contentSplitter.addWidget(self.rightPanel)

        self.rootLayout.addWidget(self.contentSplitter)

        self.footerBar = QFrame(Widget)
        self.footerBar.setObjectName(u"footerBar")
        self.footerBar.setMinimumSize(QSize(0, 34))
        self.footerBar.setMaximumSize(QSize(16777215, 34))
        self.footerLayout = QHBoxLayout(self.footerBar)
        self.footerLayout.setObjectName(u"footerLayout")
        self.footerLayout.setContentsMargins(12, 0, 12, 0)
        self.statusLabel = QLabel(self.footerBar)
        self.statusLabel.setObjectName(u"statusLabel")

        self.footerLayout.addWidget(self.statusLabel)

        self.footerSpacer = QSpacerItem(40, 20, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.footerLayout.addItem(self.footerSpacer)

        self.modelLabel = QLabel(self.footerBar)
        self.modelLabel.setObjectName(u"modelLabel")

        self.footerLayout.addWidget(self.modelLabel)


        self.rootLayout.addWidget(self.footerBar)


        self.retranslateUi(Widget)

        QMetaObject.connectSlotsByName(Widget)
    # setupUi

    def retranslateUi(self, Widget):
        Widget.setWindowTitle(QCoreApplication.translate("Widget", u"SmartCheckout \u00b7 \u667a\u80fd\u79f0\u91cd\u53f0", None))
        self.titleLabel.setText(QCoreApplication.translate("Widget", u"SmartCheckout  \u667a\u80fd\u79f0\u91cd\u53f0", None))
        self.subtitleLabel.setText(QCoreApplication.translate("Widget", u"YOLO \u5305\u88c5\u68c0\u6d4b \u00b7 ResNet \u7279\u5f81\u68c0\u7d22 \u00b7 ONNX Runtime", None))
        self.providerBadge.setText(QCoreApplication.translate("Widget", u"\u63a8\u7406\u5f15\u64ce\u5f85\u52a0\u8f7d", None))
        self.sourceCaption.setText(QCoreApplication.translate("Widget", u"\u8f93\u5165\u6e90", None))
        self.sourceCombo.setItemText(0, QCoreApplication.translate("Widget", u"\u6444\u50cf\u5934 0", None))
        self.sourceCombo.setItemText(1, QCoreApplication.translate("Widget", u"\u6444\u50cf\u5934 1", None))
        self.sourceCombo.setItemText(2, QCoreApplication.translate("Widget", u"\u6444\u50cf\u5934 2", None))
        self.sourceCombo.setItemText(3, QCoreApplication.translate("Widget", u"\u672c\u5730\u89c6\u9891", None))

        self.chooseVideoButton.setText(QCoreApplication.translate("Widget", u"\u9009\u62e9\u89c6\u9891", None))
        self.sourcePathLabel.setText(QCoreApplication.translate("Widget", u"\u5f53\u524d\uff1a\u6444\u50cf\u5934 0", None))
        self.startButton.setText(QCoreApplication.translate("Widget", u"\u25b6  \u5f00\u59cb\u8bc6\u522b", None))
        self.pauseButton.setText(QCoreApplication.translate("Widget", u"\u2161  \u6682\u505c", None))
        self.stopButton.setText(QCoreApplication.translate("Widget", u"\u25a0  \u505c\u6b62", None))
        self.liveDot.setText(QCoreApplication.translate("Widget", u"\u25cf", None))
        self.videoTitleLabel.setText(QCoreApplication.translate("Widget", u"\u5b9e\u65f6\u8bc6\u522b\u753b\u9762", None))
        self.detectionBadge.setText(QCoreApplication.translate("Widget", u"\u76ee\u6807 0", None))
        self.fpsBadge.setText(QCoreApplication.translate("Widget", u"0.0 FPS", None))
        self.videoLabel.setText(QCoreApplication.translate("Widget", u"\u9009\u62e9\u8f93\u5165\u6e90\u540e\u70b9\u51fb\u201c\u5f00\u59cb\u8bc6\u522b\u201d\\n\\n\u6a21\u578b\u5c06\u5728\u540e\u53f0\u7ebf\u7a0b\u4e2d\u52a0\u8f7d\uff0c\u754c\u9762\u4e0d\u4f1a\u5361\u4f4f", None))
        self.recognitionCaption.setText(QCoreApplication.translate("Widget", u"\u8bc6\u522b\u72b6\u6001", None))
        self.recognitionState.setText(QCoreApplication.translate("Widget", u"\u5c31\u7eea", None))
        self.thresholdCaption.setText(QCoreApplication.translate("Widget", u"\u5f00\u653e\u96c6\u9608\u503c", None))
        self.thresholdLabel.setText(QCoreApplication.translate("Widget", u"\u76f8\u4f3c\u5ea6 \u2265 0.80  \u00b7  \u95f4\u9694 \u2265 0.15", None))
        self.cartTitleLabel.setText(QCoreApplication.translate("Widget", u"\u8d2d\u7269\u8f66\u660e\u7ec6", None))
        self.cartCountBadge.setText(QCoreApplication.translate("Widget", u"0 \u4ef6", None))
        ___qtablewidgetitem = self.cartTable.horizontalHeaderItem(0)
        ___qtablewidgetitem.setText(QCoreApplication.translate("Widget", u"\u5546\u54c1", None))
        ___qtablewidgetitem1 = self.cartTable.horizontalHeaderItem(1)
        ___qtablewidgetitem1.setText(QCoreApplication.translate("Widget", u"\u5355\u4ef7", None))
        ___qtablewidgetitem2 = self.cartTable.horizontalHeaderItem(2)
        ___qtablewidgetitem2.setText(QCoreApplication.translate("Widget", u"\u6570\u91cf", None))
        ___qtablewidgetitem3 = self.cartTable.horizontalHeaderItem(3)
        ___qtablewidgetitem3.setText(QCoreApplication.translate("Widget", u"\u5c0f\u8ba1", None))
        self.unknownIcon.setText(QCoreApplication.translate("Widget", u"!", None))
        self.unknownLabel.setText(QCoreApplication.translate("Widget", u"\u68c0\u6d4b\u5230\u672a\u6ce8\u518c\u5546\u54c1\uff0c\u4e0d\u8ba1\u5165\u8d2d\u7269\u8f66", None))
        self.totalCaption.setText(QCoreApplication.translate("Widget", u"\u5e94\u4ed8\u91d1\u989d", None))
        self.totalQuantityLabel.setText(QCoreApplication.translate("Widget", u"\u5171 0 \u4ef6\u5546\u54c1", None))
        self.totalAmountLabel.setText(QCoreApplication.translate("Widget", u"\u00a50.00", None))
        self.resetButton.setText(QCoreApplication.translate("Widget", u"\u91cd\u7f6e\u8d2d\u7269\u8f66", None))
        self.settleButton.setText(QCoreApplication.translate("Widget", u"\u7ed3 \u7b97", None))
        self.statusLabel.setText(QCoreApplication.translate("Widget", u"\u7cfb\u7edf\u5c31\u7eea\uff0c\u6a21\u578b\u5c1a\u672a\u52a0\u8f7d", None))
        self.modelLabel.setText(QCoreApplication.translate("Widget", u"ONNX \u00b7 4\u5305\u88c5\u7c7b\u522b \u00b7 24 SKU", None))
    # retranslateUi

