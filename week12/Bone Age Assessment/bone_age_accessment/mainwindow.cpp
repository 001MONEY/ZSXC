#include "mainwindow.h"
#include "ui_mainwindow.h"

#include <QDebug>
#include <QFileDialog>
#include <QFileInfo>
#include <QHeaderView>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMessageBox>
#include <QNetworkReply>
#include <QPixmap>
#include <QTableWidgetItem>

// 本地推理服务地址与 Python 解释器（按实际环境修改）
static const char *kServerUrl = "http://127.0.0.1:8765/predict";
static const QString kPython = "D:/project/step1/env/python.exe";
static const QString kServerScript = "D:/project/step1/week12/Bone Age Assessment/qt_server.py";

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    net_ = new QNetworkAccessManager(this);
    connect(net_, &QNetworkAccessManager::finished,
            this, &MainWindow::onReplyFinished);
    connect(ui->btnSelect, &QPushButton::clicked,
            this, &MainWindow::onSelectImage);
    connect(ui->btnRun, &QPushButton::clicked,
            this, &MainWindow::onRunPredict);

    // 表格：行高自适应、只读
    ui->tableDetail->setEditTriggers(QAbstractItemView::NoEditTriggers);
    ui->tableDetail->horizontalHeader()->setSectionResizeMode(0, QHeaderView::ResizeToContents);
    ui->tableDetail->horizontalHeader()->setSectionResizeMode(1, QHeaderView::ResizeToContents);
    ui->tableDetail->horizontalHeader()->setSectionResizeMode(2, QHeaderView::Stretch);

    startServer();
    setStatus(QStringLiteral("就绪（推理服务启动中...）"));
}

MainWindow::~MainWindow()
{
    stopServer();
    delete ui;
}

void MainWindow::startServer()
{
    if (server_)
        return;
    server_ = new QProcess(this);
    connect(server_, &QProcess::finished,
            this, &MainWindow::onServerFinished);
    // 工作目录设为脚本所在目录，保证相对路径正确
    server_->setWorkingDirectory(QFileInfo(kServerScript).absolutePath());
    server_->start(kPython, {kServerScript, "--port", "8765"});
    qInfo() << "[server] 启动 Python 推理服务" << kServerScript;
}

void MainWindow::stopServer()
{
    if (server_ && server_->state() != QProcess::NotRunning) {
        server_->kill();
        server_->waitForFinished(3000);
    }
}

void MainWindow::onServerFinished(int exitCode, QProcess::ExitStatus status)
{
    qWarning() << "[server] 推理服务退出 code=" << exitCode << "status=" << status;
    if (server_)
        setStatus(QStringLiteral("推理服务已停止（exit %1）").arg(exitCode));
}

void MainWindow::setStatus(const QString &msg)
{
    ui->statusbar->showMessage(msg);
}

void MainWindow::onSelectImage()
{
    const QString file = QFileDialog::getOpenFileName(
        this, QStringLiteral("选择 X 光片"),
        QString(), QStringLiteral("图片文件 (*.png *.jpg *.jpeg *.bmp)"));
    if (file.isEmpty())
        return;

    imagePath_ = file;
    QPixmap pm(file);
    if (!pm.isNull())
        ui->imgInput->setPixmap(pm.scaled(ui->imgInput->size(),
                                          Qt::KeepAspectRatio,
                                          Qt::SmoothTransformation));
    setStatus(QStringLiteral("已选择: %1").arg(file));
}

void MainWindow::onRunPredict()
{
    if (imagePath_.isEmpty()) {
        QMessageBox::warning(this, QStringLiteral("提示"),
                             QStringLiteral("请先选择图片"));
        return;
    }
    if (predictBusy_) {
        setStatus(QStringLiteral("评估中，请稍候..."));
        return;
    }

    QJsonObject body;
    body["image"] = imagePath_;
    body["sex"] = ui->sexCombo->currentText();

    QNetworkRequest req(QUrl(QString::fromLatin1(kServerUrl)));
    req.setHeader(QNetworkRequest::ContentTypeHeader,
                  QStringLiteral("application/json"));
    predictBusy_ = true;
    ui->btnRun->setEnabled(false);
    setStatus(QStringLiteral("正在评估..."));
    net_->post(req, QJsonDocument(body).toJson(QJsonDocument::Compact));
}

void MainWindow::onReplyFinished(QNetworkReply *reply)
{
    predictBusy_ = false;
    ui->btnRun->setEnabled(true);
    reply->deleteLater();

    if (reply->error() != QNetworkReply::NoError) {
        setStatus(QStringLiteral("请求失败: %1").arg(reply->errorString()));
        QMessageBox::warning(this, QStringLiteral("错误"),
                             QStringLiteral("无法连接推理服务：%1\n请确认 qt_server.py 已启动。")
                                 .arg(reply->errorString()));
        return;
    }

    const QJsonDocument doc = QJsonDocument::fromJson(reply->readAll());
    if (!doc.isObject()) {
        setStatus(QStringLiteral("响应解析失败"));
        return;
    }
    const QJsonObject o = doc.object();
    if (o.contains(QStringLiteral("error"))) {
        setStatus(QStringLiteral("推理错误: %1").arg(o["error"].toString()));
        QMessageBox::warning(this, QStringLiteral("错误"), o["error"].toString());
        return;
    }

    // 骨龄
    const double years = o["bone_age_years"].toDouble();
    const double months = o["bone_age_months"].toDouble();
    ui->labelAge->setText(QStringLiteral("骨龄: %1 岁（%2 个月）")
                              .arg(years, 0, 'f', 2)
                              .arg(months, 0, 'f', 0));

    // 检出信息
    const int nBones = o["n_bones"].toInt();
    const QJsonArray missing = o["missing"].toArray();
    QString missText = missing.isEmpty()
                           ? QStringLiteral("无")
                           : QString::fromStdString(
                                 QJsonDocument(missing).toJson().toStdString());
    ui->labelInfo->setText(QStringLiteral("检出 %1/13 骨；缺失: %2")
                               .arg(nBones)
                               .arg(missText));

    // 标注图
    const QString visPath = o["vis_path"].toString();
    QPixmap vis(visPath);
    if (!vis.isNull())
        ui->imgOutput->setPixmap(vis.scaled(ui->imgOutput->size(),
                                            Qt::KeepAspectRatio,
                                            Qt::SmoothTransformation));

    // 13 骨明细表
    const QJsonArray detail = o["detail"].toArray();
    ui->tableDetail->setRowCount(detail.size());
    for (int i = 0; i < detail.size(); ++i) {
        const QJsonObject d = detail[i].toObject();
        const QString bone = d["bone"].toString();
        const QString grade = d["grade"].isNull()
                                  ? QStringLiteral("-")
                                  : QString::number(d["grade"].toInt());
        const QString score = d["score"].isNull()
                                  ? QStringLiteral("-")
                                  : QString::number(d["score"].toDouble(), 'f', 1);
        ui->tableDetail->setItem(i, 0, new QTableWidgetItem(bone));
        ui->tableDetail->setItem(i, 1, new QTableWidgetItem(grade));
        ui->tableDetail->setItem(i, 2, new QTableWidgetItem(score));
    }

    setStatus(QStringLiteral("评估完成"));
}
