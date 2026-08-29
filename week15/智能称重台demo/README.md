# SmartCheckout 智能称重台

基于 YOLO、ResNet 特征检索、MySQL、ONNX Runtime 和 PySide6 的桌面端智能商品识别与结算系统。

> 当前答辩演示初始状态：**24 个训练 SKU、阿萨姆奶茶未注册**；现场注册后无需重训即可变为 25 SKU。系统使用 ONNX CUDA GPU 推理、开放集未注册商品拦截以及 Qt 购物车与结算界面。  
> 模型冻结日期：2026-08-27；在线注册增强日期：2026-08-28。详细约束见 [MODEL_FREEZE.md](MODEL_FREEZE.md)。

## 1. 项目目标

系统从摄像头或本地视频读取画面，完成以下闭环：

1. YOLO 检测袋装、瓶装、盒装和罐装商品。
2. 按包装大类选择对应的 ResNet18 特征模型。
3. 从商品裁剪图提取 512 维 L2 归一化特征。
4. 在对应大类的特征库中进行类中心与 Top-K 样本联合检索。
5. 使用相似度和 Top1/Top2 间隔进行开放集判定，拒绝未注册商品。
6. 从 MySQL 查询商品名称和单价，并通过 25 帧稳定窗口生成购物车。
7. 在 Qt 界面中显示识别画面、购物车、金额、结算和商品注册入口。

## 2. 最终实现状态

| 模块      | 当前状态                                         |
| ------- | -------------------------------------------- |
| YOLO 检测 | 4 类：`bag`、`bottle`、`box`、`cylinder`          |
| SKU 识别  | 初始24个训练 SKU；现场注册阿萨姆后变为25个                  |
| 特征模型    | 4 个 ResNet18，去掉分类头后输出 512 维特征                |
| 特征检索    | NumPy 向量矩阵，`0.7 × 类中心 + 0.3 × 同类 Top5 均值`    |
| 开放集判定   | 原24类使用 `0.80/0.15`；在线注册类使用高相似度 `0.95/0.01` |
| 商品数据库   | MySQL `smart_checkout.products`，演示初始24条、注册后25条 |
| 部署格式    | 1 个 YOLO ONNX + 4 个 ResNet ONNX              |
| 推理设备    | 优先 `CUDAExecutionProvider`，不可用时回退 CPU        |
| 桌面界面    | PySide6/Qt，支持摄像头、本地视频、暂停、结算、重置和注册入口          |
| 答辩验证    | 注册前7件、¥38.60；现场注册后同一视频8件、¥41.60             |

训练结果摘要：

| 模型                | 最佳验证指标                                                      |
| ----------------- | -----------------------------------------------------------:|
| YOLOv8n 检测        | Precision 0.9946、Recall 0.9868、mAP50 0.9892、mAP50-95 0.7883 |
| bag ResNet18      | Top-1 100%                                                  |
| bottle ResNet18   | Top-1 100%                                                  |
| box ResNet18      | Top-1 100%                                                  |
| cylinder ResNet18 | Top-1 99.59%                                                |

> 分类验证集来自同一项目的视频采样，指标用于项目内部检查；真实摄像头效果还会受到背景、反光、遮挡、姿态和域差异影响。

## 3. 系统架构

```text
摄像头 / 本地视频
        │
        ▼
YOLO ONNX：包装检测（4类）
        │ 检测框 + package_type
        ▼
裁剪、5% padding、最小框过滤
        │
        ▼
对应大类 ResNet18 ONNX：512维特征
        │
        ▼
特征库检索：类中心 + 同类 Top5 样本
        │
        ├─ 原24类：sim < 0.80 或 margin < 0.15 ──► 未注册
        ├─ 在线类：sim < 0.95 或 margin < 0.01 ──► 不接受动态类
        │
        └─ 通过开放集阈值
                 │
                 ▼
        MySQL products 查询名称和价格
                 │
                 ▼
        25帧稳定窗口 → 购物车 → Qt结算界面
```

核心检索分数：

```text
score(class) = 0.7 × cosine(query, class_center)
             + 0.3 × mean(top5 cosine(query, class_samples))

margin = top1_score - top2_score
```

## 4. 目录结构

```text
智能称重台demo/
├─ README.md                              项目总入口文档
├─ 项目总结.md                           开发总结、困难、踩坑和未来改进
├─ MODEL_FREEZE.md                       答辩基线、阈值和模型冻结说明
├─ smart_checkout_qt/                    最终答辩版 Qt 工程
├─ smart_checkout_ui/                    另一套历史 Qt 实现，保留作对照
├─ database/                             MySQL 连接和商品 DAO
├─ video/                                原始训练、验证和补充视频，不提交 Git
├─ yolo_dataset_raw/                     YOLO 图片及标签数据集
├─ classification_dataset_from_videos/  24 SKU 分类训练/验证数据集
├─ runs/
│  ├─ detect/                            YOLO 训练结果
│  ├─ classify/                          4 个 ResNet18 训练结果
│  ├─ features/                          4 个特征向量库
│  ├─ onnx/                              5 个 ONNX 模型
│  └─ pipeline/                          端到端验证视频、CSV 和 JSON
├─ unknown_samples/                      未注册商品阈值标定样本
├─ work/                                 数据检查、问题分析和一次性维护脚本
├─ 商品数据.xlsx                         24 SKU 商品原始信息
└─ 智能称重台项目计划书.md               项目计划书
```

数据集、模型和视频体积较大，已被 `.gitignore` 排除。迁移项目时必须单独复制 `video/`、`runs/`、数据集和 MySQL 数据。

## 5. 环境与依赖

推荐解释器：

```text
D:\project\step1\env\python.exe
Python 3.11
```

Qt 运行依赖记录在 [smart_checkout_qt/requirements.txt](smart_checkout_qt/requirements.txt)：

```powershell
D:\project\step1\env\python.exe -m pip install -r smart_checkout_qt\requirements.txt
```

训练和数据处理还需要：

- PyTorch、torchvision
- OpenCV、Pillow、NumPy
- PyYAML、PyMySQL
- Ultralytics 8.4.113（仅重训 YOLO 或重新导出 YOLO 时需要）
- ONNX、onnxruntime-gpu
- PySide6

MySQL 默认配置位于 [database/mysql_db.py](database/mysql_db.py)。当前为本机教学环境配置，迁移或发布前应改为环境变量，不要在公开仓库中保存真实密码。

> **GPU 加速依赖说明**：`onnx_engine.py` 通过把 PyTorch 自带的 CUDA/cuDNN DLL（`torch/lib` 下的 `cublasLt64_12.dll`、`cudnn64_9.dll`、`cudart64_12.dll`）注入 DLL 搜索路径来启用 `CUDAExecutionProvider`。因此 GPU 加速**依赖环境中存在 PyTorch**，只安装 `onnxruntime-gpu` 不够；没有 CUDA 环境时自动回退 CPU。

## 6. 快速启动

> **前提**：确保本机 MySQL 服务已启动（`smart_checkout` 库与 `products` 表存在，连接信息见 `database/mysql_db.py`）。推理/结算会实时查库，MySQL 未启动会导致启动或识别阶段报错。

### 6.1 最终 Qt 答辩界面

```powershell
cd D:\project\step1\week15\智能称重台demo
D:\project\step1\env\python.exe smart_checkout_qt\widget.py
```

Qt Creator 中的解释器选择：

```text
D:\project\step1\env\python.exe
```

### 6.2 命令行端到端视频推理

```powershell
D:\project\step1\env\python.exe pipeline_demo.py `
  --source "video\YOLO Data\val\VID_20260826_110333.mp4" `
  --name final_verify
```

### 6.3 摄像头实时推理

```powershell
D:\project\step1\env\python.exe pipeline_demo.py --camera 0
```

摄像头窗口快捷键：`S` 结算、`R` 重置、`Esc` 退出。

### 6.4 无摄像头回归测试

```powershell
D:\project\step1\env\python.exe smart_checkout_qt\smoke_test.py
D:\project\step1\env\python.exe smart_checkout_qt\annotation_boundary_test.py
D:\project\step1\env\python.exe smart_checkout_qt\engine_smoke_test.py
D:\project\step1\env\python.exe smart_checkout_qt\registration_smoke_test.py
```

## 7. 冻结参数

答辩前不要随意修改以下参数：

| 参数                   | 值     |
| -------------------- | -----:|
| YOLO 输入尺寸            | 640   |
| YOLO confidence      | 0.25  |
| YOLO IoU             | 0.45  |
| 裁剪 padding           | 0.05  |
| 最小检测框                | 24 px |
| similarity threshold | 0.80  |
| margin threshold     | 0.15  |
| 在线注册类 similarity | 0.95  |
| 在线注册类 margin     | 0.01  |
| 稳定窗口                 | 25 帧  |

原24类阈值保持冻结，不通过全局降阈值换取新商品命中。在线注册类必须先达到更高的 `0.95` 相似度，才允许使用较小类别间隔；动态类未通过时会退回原24类判定，避免影响橙汁等旧商品。

## 8. 脚本说明

### 8.1 YOLO 数据准备与检测

| 文件                         | 作用                                                                |
| -------------------------- | ----------------------------------------------------------------- |
| `extract_yolo_frames.py`   | 从 YOLO 训练/验证视频按固定帧率抽图，创建标准 `images/labels` 目录和抽帧清单。               |
| `append_yolo_frames.py`    | 根据抽帧清单只处理后来新增的视频，并从已有编号继续命名，避免重复抽取。                               |
| `rename_yolo_images.py`    | 安全缩短 YOLO 图片文件名，并同步重命名同名标签；默认预览，`--apply` 才执行。                    |
| `check_yolo_labels.py`     | 检查图像/标签对应关系、YOLO 行格式、归一化范围、类别编号、空标签和重复框。                          |
| `train_yolov8.py`          | 训练4类 YOLOv8n 检测模型；支持 `--check-only`，正式训练需项目内 Ultralytics 8.4.113。 |
| `run_train_yolov8.bat`     | 使用固定 Python 3.11 环境调用 `train_yolov8.py` 的 Windows 快捷入口。           |
| `infer_video.py`           | 对指定视频或图片目录执行 YOLO 检测推理，输出带框结果。                                    |
| `smart_checkout_data.yaml` | YOLO 数据集路径和4个包装类别定义。                                              |

### 8.2 分类数据集与 ResNet 训练

| 文件                                            | 作用                                                                              |
| --------------------------------------------- | ------------------------------------------------------------------------------- |
| `extract_frames.py`                           | 从按 SKU 组织的商品视频按时间间隔抽图，生成 ImageFolder 结构的原始分类数据集。                                |
| `rename_frames.py`                            | 将人工筛选后的分类图片安全改为连续短文件名，并生成 CSV 映射。                                               |
| `crop_classification_dataset.py`              | 使用训练好的 YOLO 框裁切已有分类图片，保留24个 SKU 目录。                                             |
| `build_classification_dataset_from_videos.py` | 直接从 SKU 视频完成 YOLO 定位、目标跟踪、裁切、清晰度过滤、感知哈希去重和数据清单生成；SKU 标签继承视频目录。                  |
| `train_resnet_classifier.py`                  | 分别训练 bag/bottle/box/cylinder 四个6分类 ResNet18；不带 `--train` 只检查，带 `--train` 才正式训练。 |

### 8.3 特征库、开放集识别与在线注册

| 文件                           | 作用                                                                       |
| ---------------------------- | ------------------------------------------------------------------------ |
| `build_feature_library.py`   | 从训练图片和4个 ResNet18 构建 embeddings、labels、centers、classes、metadata 和 stats。 |
| `augment_feature_library.py` | 用某个已注册 SKU 的补充真实场景视频追加特征样本，不重训模型；星巴克域适配使用过该脚本。                           |
| `calibrate_threshold.py`     | 结合注册商品验证图和未注册商品照片统计 Top1 相似度与类别间隔，辅助确定开放集阈值。                             |
| `feature_library_updater.py` | Qt 注册后端：视频均匀抽样、清晰度过滤、感知哈希去重、多姿态原型、MySQL 与特征文件一致更新。             |
| `register_from_video.py`     | 多姿态视频注册/诊断命令行入口；不带 `--apply` 只读分析，带 `--apply` 才写入并生成验证报告。              |
| `prepare_registration_demo.py` | 录制前备份当前状态并恢复24-SKU初始基线；默认只读检查，带 `--reset` 才执行恢复。                    |
| `register_goods.py`          | MySQL 商品的命令行增删改查工具；只改商品表，不自动补充视觉特征。                                      |

当前特征库不是 FAISS，而是 NumPy 数组加余弦相似度。24～25 SKU 规模下可直接加载到内存，后续规模扩大时再考虑 FAISS。

### 8.4 ONNX 与端到端流水线

| 文件                 | 作用                                                               |
| ------------------ | ---------------------------------------------------------------- |
| `export_onnx.py`   | 导出 YOLO 检测模型和4个去分类头的 ResNet18 特征模型。                              |
| `verify_onnx.py`   | 对比 PT 与 ONNX 的特征余弦、检测类别、置信度和框 IoU，防止导出后结果漂移。                     |
| `onnx_engine.py`   | ONNX Runtime 推理核心，包含 letterbox、YOLO 解码/NMS、特征预处理和向量检索；自动优先 CUDA。 |
| `pipeline_demo.py` | 命令行端到端入口：检测、检索、开放集判定、MySQL 查价、稳定计数、视频/摄像头显示和结果落盘。                |
| `MODEL_FREEZE.md`  | 记录最终权重、特征库、阈值、验证证据和 ONNX 部署约束。                                   |

### 8.5 数据库模块

| 文件                      | 作用                                                   |
| ----------------------- | ---------------------------------------------------- |
| `database/mysql_db.py`  | PyMySQL 连接、查询和执行语句的轻量封装。                             |
| `database/goods_dao.py` | `products` 表的数据访问层，按 SKU、模型类别、条码查询，支持注册、更新、删除和购物车汇总。 |
| `database/__init__.py`  | 数据库包声明。                                              |
| `商品数据.xlsx`             | 24个答辩 SKU 的名称、包装类型、价格等原始录入数据。                        |

### 8.6 最终 Qt 工程 `smart_checkout_qt/`

| 文件                            | 作用                                                             |
| ----------------------------- | -------------------------------------------------------------- |
| `widget.py`                   | 最终主窗口入口，负责输入源、按钮状态、视频显示、购物车、告警和结算交互。                           |
| `form.ui`                     | Qt Designer 可编辑界面文件。                                           |
| `ui_form.py`                  | 由 `pyside6-uic form.ui -o ui_form.py` 生成的 Python 界面代码，不建议手工修改。 |
| `inference_controller.py`     | QThread 后台推理控制器，负责 ONNX、特征检索、开放集判定、未知样本缓存、稳定购物车和画框。            |
| `registration_dialog.py`      | 未注册商品录入弹窗，填写名称、价格、SKU 和检索分类名，并调用增量特征后端。                        |
| `smoke_test.py`               | 离屏创建主窗口，检查关键控件存在且不会自动启动摄像头。                                    |
| `engine_smoke_test.py`        | 读取固定验证视频的一帧，检查 ONNX、CUDA/CPU Provider、特征库和 MySQL 链路。           |
| `annotation_boundary_test.py` | 验证检测框超出图像四个方向时仍可安全绘制，防止 Pillow 反向矩形异常。                         |
| `registration_smoke_test.py`  | 抽取阿萨姆竖放/横放帧，验证跨组动态检索、立即计价及同帧重复框去重。                          |
| `pyproject.toml`              | Qt 工程元数据、Python版本和依赖声明。                                        |
| `requirements.txt`            | Qt 运行依赖。                                                       |
| `README.md`                   | Qt 子工程启动说明。                                                    |

### 8.7 历史对照 Qt 工程 `smart_checkout_ui/`

该目录是另一套独立实现，不是最终答辩入口，保留用于功能对照。

| 文件                    | 作用                        |
| --------------------- | ------------------------- |
| `main.py`             | 历史版 QMainWindow 界面及注册弹窗。  |
| `inference_worker.py` | 历史版 Qt 推理线程、未知样本缓存和热重载逻辑。 |
| `smoke_test.py`       | 历史版界面与单帧推理冒烟测试。           |
| `README.md`           | 历史版运行说明。                  |

### 8.8 分析、复核和一次性维护脚本

以下脚本用于开发过程中的问题定位或一次性数据维护，不是日常启动入口；部分脚本包含固定路径或依赖历史输出目录。

| 文件                                 | 作用                                                   |
| ---------------------------------- | ---------------------------------------------------- |
| `_analyze_coke.py`                 | 历史可口可乐视频分析，统计未注册结果并保存少量裁剪样本。                         |
| `_test_register.py`                | 注册功能闭环测试：临时备份、注册测试商品、验证并恢复；会短暂写数据库和特征库。              |
| `work/analyze_conf.py`             | 分析旧流水线 CSV 中的分类置信度分布。                                |
| `work/analyze_retrieval.py`        | 分析 retrieval_v2 的 Top1、Top2、margin 和星巴克记录。           |
| `work/analyze_v3.py`               | 分析 retrieval_v3 中星巴克相关排名。                            |
| `work/build_val_review_sheets.py`  | 从分类 val 清单生成每个 SKU 的人工复核拼图。                          |
| `work/check_mysql.py`              | 早期 MySQL 状态检查脚本，仍引用 week06 路径；当前优先使用项目内 `database/`。 |
| `work/extract_bottle_check.py`     | 提取指定验证帧中的瓶装裁剪，用于人工确认阿萨姆/星巴克问题。                       |
| `work/inspect_imported_videos.py`  | 检查新导入视频的时长、亮度、清晰度并生成代表帧拼图。                           |
| `work/inspect_new_yolo_videos.py`  | 找出抽帧清单中尚未处理的 YOLO 视频并生成检查报告。                         |
| `work/inspect_remaining_videos.py` | 为剩余瓶装分类视频生成多时间点检查拼图。                                 |
| `work/inspect_soft_clips.py`       | 对两个历史偏糊视频生成细粒度清晰度拼图。                                 |
| `work/review_yolo_duplicates.py`   | 生成 YOLO 训练图总览页并计算相邻帧差异，辅助人工复核重复帧。                    |
| `work/prune_yolo_duplicates.py`    | 将人工选定的近重复 YOLO 图片及标签移动到可恢复目录；默认只预览，`--apply` 才执行。    |
| `work/remove_test_skus.py`         | 一次性删除阿萨姆和有糖可乐测试 SKU，并备份特征与数据库记录；当前已执行，不应重复运行。        |

## 9. 主要输出文件

| 路径                                                              | 内容                            |
| --------------------------------------------------------------- | ----------------------------- |
| `runs/detect/smart_checkout_yolov8n/weights/best.pt`            | 最佳 YOLO PT 权重                 |
| `runs/classify/<group>_resnet18/best.pt`                        | 4个最佳 ResNet18 权重              |
| `runs/features/<group>_*`                                       | 4个包装大类的特征库与元数据                |
| `runs/onnx/yolov8n_det.onnx`                                    | YOLO ONNX 模型                  |
| `runs/onnx/<group>_resnet18_feat.onnx`                          | 4个特征 ONNX 模型                  |
| `runs/pipeline/final_verify/`                                   | 7件 ¥38.60 的最终端到端验证证据          |
| `runs/pipeline/demo_stage1_unregistered/`                       | 当前24-SKU初始态复验：7件 ¥38.60，阿萨姆未计价 |
| `runs/pipeline/asm_registered_verify_v2/`                       | 注册后独立视频：8件 ¥41.60，原7件无回归     |
| `runs/pipeline/asm_new_video_verify_v2/`                        | 新拍视频：阿萨姆1件 ¥3.00，391/393帧稳定      |
| `runs/registration/bottle07_20260828_final_verification.json`   | 在线注册抽样、原型与两组验证结果汇总           |
| `runs/pipeline/starbucks_verify/`                               | 星巴克真实场景补充后的验证证据               |
| `runs/pipeline/onnx_e2e/`、`onnx_gpu/`                           | ONNX 端到端与 GPU 加速验证（结果与 PT 一致） |
| `runs/pipeline/camera_test/`、`camera_unknown2/`、`camera_final/` | 摄像头实测证据：已注册识别、未注册拒绝、0 件结算     |
| `work/sku_cleanup_backups/20260827_220011/`                     | 删除两个临时测试 SKU 前的可恢复备份          |

## 10. 答辩演示建议

录制前先关闭正在运行的 Qt 程序，在项目根目录执行：

```powershell
D:\project\step1\env\python.exe prepare_registration_demo.py --reset
```

该命令会先备份当前 bottle 特征库和 MySQL 商品表，再恢复到可重复录制的24-SKU初始状态。演示顺序如下：

1. 启动 `smart_checkout_qt/widget.py`，确认右下角显示 `24 SKU`，Provider 显示 `CUDA GPU`。
2. 选择 `video/YOLO Data/val/VID_20260826_110333.mp4` 并开始识别：阿萨姆显示“未注册”，购物车最终应为原7件、¥38.60。
3. 停止识别，切换到 `video/asm milktea.mp4` 并开始播放；出现未注册告警后点击“注册商品”。
4. 在注册弹窗中把“实际包装类型”改为“瓶装”，填写名称 `阿萨姆奶茶`、价格 `3.00`、SKU `bottle07`、检索分类名 `BOTTLE_07_asm milktea`。弹窗会自动使用当前补充视频。
5. 点击“确认注册”，等待成功提示和特征库热重载；主界面右下角应变为 `25 SKU`。
6. 停止并重新选择第2步的验证视频，再识别一次：购物车应为原7件加阿萨姆，共8件、¥41.60。
7. 讲解时说明：注册没有重训 ResNet，而是从补充视频跨时间均匀抽样、过滤模糊与重复帧、建立多姿态原型，然后同步写入 MySQL 并热重载。
8. 回答阈值问题时强调：原24类始终使用0.80/0.15；新增类使用高相似度0.95作为安全前提，没有降低全局阈值。

若录制中途失败，关闭 Qt 后重新执行同一条 `--reset` 命令即可从头再录。

## 11. 已知限制

- 复杂纹理背景可能产生 YOLO 假阳性，长期需要空场景和桌布等困难负样本。
- 横放罐装商品在当前 YOLO 数据中覆盖不足，可能直接漏检。
- 在线注册视频应只包含一件目标商品；多件同包装未知商品仍需要目标跟踪后分别注册。
- 普通交叉熵训练的 ResNet 特征空间没有显式优化角度间隔；有糖/无糖可乐等近似包装仍是难点。
- 原24类使用单类中心；在线注册类已支持最多4个姿态原型。
- 当前没有使用 FAISS、ArcFace 或 Web 端，这些属于后续可选加分方向。

更完整的开发过程、困难和改进方向见 [项目总结.md](项目总结.md)。
