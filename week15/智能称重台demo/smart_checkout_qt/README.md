# SmartCheckout Qt

智能称重台的 PySide6 桌面界面。此目录是独立的 Qt Creator 工程，不依赖
`smart_checkout_ui` 中的代码。

## 启动

在 Qt Creator 中直接运行 `widget.py`，解释器使用：

```text
D:\project\step1\env\python.exe
```

也可以在项目根目录执行：

```powershell
D:\project\step1\env\python.exe smart_checkout_qt\widget.py
```

## 工作流

1. 选择摄像头，或切换到“本地视频”后选择文件。
2. 点击“开始识别”。首次启动会在后台加载 5 个 ONNX 会话。
3. 画面左侧显示检测框，右侧使用 25 帧稳定窗口显示购物车。
4. 原24类使用 `0.80/0.15` 开放集阈值；在线注册类使用高相似度 `0.95` 和专用间隔 `0.01`。
5. “暂停”保留当前购物车，“停止”释放输入源，“重置购物车”清空稳定窗口。

## 注册未注册商品（操作步骤）

1. 把未注册商品放到摄像头前（或选择含该商品的视频），等画面出现红色告警卡片。
2. 点击告警卡片中的“注册商品”，弹出录入窗口。
3. 确认实际包装类型，填写商品名称、单价（SKU 与分类名会自动建议，可修改）。
4. 若主界面选择了本地视频，弹窗会自动把它作为多姿态注册视频；也可以手动更换。
5. 点击“确认注册”：程序会均匀抽帧、过滤模糊和重复裁剪、建立多姿态原型，并在推理线程内热重载。
6. 下一帧起该商品即可识别、进购物车并结算，注册前的稳定窗口会自动清空。

> 提示：补充视频只拍一件商品，并覆盖正面、侧面、远近和横竖姿态。若 YOLO 对新包装分组摇摆，弹窗可改成真实包装类型，运行期会对在线注册类做跨组补充检索。

### 阿萨姆答辩录制

录制前先关闭本程序，并在项目根目录执行
`D:\project\step1\env\python.exe prepare_registration_demo.py --reset`。启动后应显示24 SKU。先播放验证视频得到7件、¥38.60，再切换到 `video/asm milktea.mp4`；出现告警后将实际包装类型选为“瓶装”，填写 `阿萨姆奶茶 / 3.00 / bottle07 / BOTTLE_07_asm milktea` 并确认注册。热重载完成后重新播放验证视频，应得到8件、¥41.60。

## 文件说明

- `form.ui`：Qt Designer 可编辑界面。
- `ui_form.py`：由 `pyside6-uic form.ui -o ui_form.py` 生成。
- `widget.py`：界面交互与信号槽。
- `inference_controller.py`：后台 ONNX 推理、开放集判定和购物车稳定逻辑。
- `registration_dialog.py`：未注册商品录入弹窗，调用增量特征后端。
- `registration_smoke_test.py`：验证阿萨姆竖放、横放、跨包装分组及单件计数。

正常识别只读取 `runs/onnx`、`runs/features` 和 MySQL 商品库；**注册功能**会按用户操作写入数据库与特征库。
