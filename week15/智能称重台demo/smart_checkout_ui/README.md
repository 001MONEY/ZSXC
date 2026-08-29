# 智能称重台 Qt 展示界面（PySide6）

计划书核心交付项：摄像头/视频切换 + 实时识别画面 + 商品列表 + 金额 + 结算按钮。

## 运行

```powershell
D:\project\step1\env\python.exe main.py
```

依赖（仅新增 PySide6，其余复用项目冻结环境）：
```powershell
D:\project\step1\env\python.exe -m pip install PySide6 -i https://mirrors.aliyun.com/pypi/simple/
```

## 功能

| 功能 | 说明 |
|---|---|
| 输入源 | 摄像头 0/1/2 或本地视频文件（自动循环播放） |
| 实时画面 | 带检测框 + 商品名 + 价格标注（复用 `pipeline_demo.annotate_frame`） |
| 购物车 | 右侧表格：商品 / 单价 / 数量 / 小计，滑动窗口（25帧）稳定组合 |
| 合计金额 | 实时更新，大字显示 |
| 未注册提示 | 红色提示最近未注册商品（相似度/间隔不达标），**不计入购物车** |
| 结算 | 弹出结算单（明细 + 总件数 + 总金额） |
| 重置 | 清空购物车 |

## 架构

```
main.py (QMainWindow UI)
   └─ InferenceWorker (QThread)
        ├─ onnx_engine.YoloOnnxDetector      ONNX GPU 检测（~7ms）
        ├─ onnx_engine.OnnxFeatureLibrary    4 个特征库 + ONNX 特征
        ├─ onnx_engine.retrieval_match_onnx  检索（类中心+TopK，阈值 sim>=0.80 / margin>=0.15）
        ├─ database.goods_dao.GoodsDao       MySQL 商品库查价/结算
        └─ pipeline_demo.annotate_frame      画框标注
```

冻结方案的阈值/特征库/预处理全部复用，`MODEL_FREEZE.md` 中约定的约束依旧生效：
**不要重建 runs/features、不要调阈值、不要重训 4 个 ResNet。**

## 说明

- 推理在线程中执行，UI 不卡顿；FPS 显示在状态栏。
- 摄像头分辨率 640x480（硬件限制），画面按比例缩放显示。
- 若提示 CUDA 加载失败，onnx_engine 会自动回退 CPU（速度较慢）。
