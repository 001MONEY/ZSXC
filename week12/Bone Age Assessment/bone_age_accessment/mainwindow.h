#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QNetworkAccessManager>
#include <QProcess>

QT_BEGIN_NAMESPACE
namespace Ui {
class MainWindow;
}
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = nullptr);
    ~MainWindow() override;

private slots:
    void onSelectImage();
    void onRunPredict();
    void onReplyFinished(QNetworkReply *reply);
    void onServerFinished(int exitCode, QProcess::ExitStatus status);

private:
    void startServer();
    void stopServer();
    void setStatus(const QString &msg);

    Ui::MainWindow *ui;
    QProcess *server_ = nullptr;
    QNetworkAccessManager *net_ = nullptr;
    QString imagePath_;
    bool predictBusy_ = false;
};
#endif // MAINWINDOW_H
