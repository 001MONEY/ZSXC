# 🔒 模型冻结快照（MODEL FREEZE）

> **模型冻结日期**：2026-08-27；**在线注册增强**：2026-08-28
> **冻结原因**：特征向量检索方案已通过视频 + 摄像头实测验证（开放集识别初步达成），
> 进入 Qt 界面开发阶段。**开发 Qt 期间冻结以下所有配置，不要重建特征库、不要调阈值。**
> **解冻条件**：如需引入新商品/重训特征模型（如 ArcFace），需重新走"标定 → 验证"流程。

---

## 1. 冻结清单

### 1.1 四个 ResNet18 分类权重（SKU 特征提取骨干）
| 包装类 | 权重路径 | 训练完成时间 | 说明 |
|---|---|---|---|
| bag | `runs/classify/bag_resnet18/best.pt` | 2026-08-27 14:48 | 6 SKU |
| bottle | `runs/classify/bottle_resnet18/best.pt` | 2026-08-27 14:49 | 6 SKU |
| box | `runs/classify/box_resnet18/best.pt` | 2026-08-27 14:51 | 6 SKU |
| cylinder | `runs/classify/cylinder_resnet18/best.pt` | 2026-08-27 15:02 | 6 SKU |

- 架构：ResNet18（ImageNet 预训练微调，40 epochs，前 3 轮冻结骨干，label_smoothing 0.1）
- 用法：`model.fc = nn.Identity()` → 输出 512 维特征向量

### 1.2 特征库（runs/features/）
| 文件 | 内容 |
|---|---|
| `{group}_embeddings.npy` | 全样本 512 维特征（L2 归一化） |
| `{group}_labels.json` | 样本 → SKU 标签 |
| `{group}_centers.npy` | 每 SKU 类中心（L2 归一化） |
| `{group}_classes.json` | 类名列表（与 products.model_class 对应） |
| `{group}_metadata.json` | 模型/预处理/版本元数据 |
| `{group}_stats.json` | 每类样本数统计 |

- bottle 已扩充到 1329 样本（含星巴克 211 个真实场景补充样本）
- 数据库 `products.feature_index` 存类中心标识（`lib1_center0_SKU`），不存向量

### 1.3 阈值（pipeline_demo.py 默认值，已标定）
| 参数 | 值 | 判定 |
|---|---|---|
| `--similarity-threshold` | **0.80** | Top1 相似度 < 0.80 → 未注册 |
| `--margin-threshold` | **0.15** | Top1−Top2 类别间隔 < 0.15 → 未注册（"跟谁都像"） |
| `--unknown-threshold` | 0.5 | 仅 classify 模式备用，默认走 retrieval |
| `--mode` | retrieval | 特征向量检索（默认） |

**开放集判定**：`sim ≥ 0.80 且 margin ≥ 0.15` → 已注册；否则未注册。
（margin 是关键——未注册商品 Top1/Top2 接近，single 相似度阈值无法区分）

### 1.4 预处理与裁切参数（冻结）
| 参数 | 值 |
|---|---|
| YOLO 推理尺寸 `--imgsz` | 640 |
| YOLO 置信度 `--conf` | 0.25 |
| YOLO NMS `--iou` | 0.45 |
| 裁切扩展 `--padding` | 0.05（5% padding） |
| 最小检测框 `--min-box-size` | 24 px |
| 摄像头显示缩放 `--display-scale` | 1.5 |
| 滑动窗口 | 25 帧稳定组合 |

---

## 2. 验证记录（三类，请保留勿删）

### ✅ 已注册星巴克（真实场景补充样本后识别成功）
- 位置：`runs/pipeline/starbucks_verify/`（result.json + result.mp4 + frame_records.csv）
- 结果：`BOTTLE_06_starbucks`，**¥7.50 × 1**，识别成功

### ✅ 阿萨姆奶茶：注册前正确拒绝，注册后立即识别
- 证据：`runs/pipeline/final_verify/frame_records.csv`
- 注册前：测试视频中阿萨姆因 margin<0.15 被拒绝，不进入购物车。
- 当前交付/录制初始状态保持在注册前24-SKU基线；执行 `prepare_registration_demo.py --reset` 可重复恢复该状态。
- 注册来源：`video/asm milktea.mp4`，66个均匀采样点，筛选52条特征，建立4个姿态原型。
- 动态类门限：`similarity >= 0.95` 且 `margin >= 0.01`；原24类仍保持0.80/0.15。
- 注册后：新视频稳定为1件、¥3.00；独立多商品视频为8件、¥41.60，胖东来橙汁等原7件均保留。

### ✅ 未注册冰红茶（不再误判为橙汁）
- 结果：摄像头实测判 "bottle 未注册"，**不再误判为胖东来橙汁**（旧 softmax 方案的错误）

### 📋 其他辅助验证记录
| 目录 | 内容 |
|---|---|
| `runs/pipeline/camera_test` | 摄像头实测：胖东来橙汁 ¥13.90 ✅ |
| `runs/pipeline/camera_unknown2` | 摄像头实测：橙汁×2 + 百事 ¥30.80，未注册被排除 |
| `runs/pipeline/final_verify` | 视频端到端：**7 件已注册 ¥38.60** 全部正确 |
| `runs/pipeline/demo_stage1_unregistered` | 24-SKU录制初始态复验：**7件 ¥38.60**，阿萨姆未计价 |
| `runs/pipeline/asm_registered_verify_v2` | 在线注册后独立视频：**8件 ¥41.60**，原7件无回归 |
| `runs/pipeline/asm_new_video_verify_v2` | 新拍阿萨姆视频：稳定 **1件 ¥3.00**，391/393帧命中最终组合 |
| `unknown_samples/`（用户拍摄） | 12 张未注册照片，用于阈值标定 |

---

## 3. Qt 界面开发复用指引

直接复用 `pipeline_demo.py` 中的现成模块（参数已固化为默认值）：
```python
from pipeline_demo import (
    load_yolo,          # YOLO 检测
    load_feature_library,  # 加载 4 个特征库（retrieval 模式）
    retrieval_match,    # 检索匹配：返回 (top1, sim, top2, sim2, margin)
    expand_box,         # 裁切扩展
    YOLO_MODEL,
)
from database.goods_dao import GoodsDao  # 商品库查询/结算
```

推荐启动参数（Qt 内调用等价逻辑）：
```bash
python pipeline_demo.py --mode retrieval --similarity-threshold 0.80 --margin-threshold 0.15
```

**约束**：开发 Qt 期间不要重建 `runs/features`、不要修改阈值默认值、不要重新训练 4 个 ResNet。
如需改动，先在此文档登记并走"标定 → 验证"闭环。

---

## 4. ONNX 导出（2026-08-27，推理引擎可切换）

### 4.1 导出产物（runs/onnx/，均由脚本重建）
| 模型 | 文件 | 输入 | 输出 |
|---|---|---|---|
| 检测 | `yolov8n_det.onnx` | 1x3x640x640 | 1x8x8400（4类，box 为**像素坐标**） |
| 特征 | `{group}_resnet18_feat.onnx` | 1x3x224x224 | 1x512（去 fc 层） |

- 导出：`D:\project\step1\env\python.exe export_onnx.py`
- 推理引擎：`onnx_engine.py`（YoloOnnxDetector + OnnxFeatureLibrary + retrieval_match_onnx）
- **一致性已验证**：`verify_onnx.py` → 特征余弦=1.000000，检测 IoU=0.9932、类别 100%、置信度差 0.0033

### 4.2 使用方式
```bash
# 默认引擎已是 ONNX（GPU优先），可直接运行：
python pipeline_demo.py --source 视频.mp4 --name xxx   # 视频模式
python pipeline_demo.py --camera 0                     # 摄像头模式
# 显式指定引擎（pt 需 third_party 源码）：
python pipeline_demo.py --engine pt --camera 0
# 验证一致性
python verify_onnx.py
```
- ONNX 端到端结果与 PT 一致：`runs/pipeline/onnx_e2e/`（7件 ¥38.60）
- 阈值/特征库不变，`--engine pt|onnx` 可随时切换
- 注意：ultralytics 导出的 YOLOv8 ONNX box 是像素坐标（非归一化），解码时勿再乘 input_size

### 4.2b GPU 加速（已启用，无需手动配置）
- `onnx_engine.py` 启动时自动将 `torch/lib`（含 cublasLt64_12.dll / cudnn64_9.dll / cudart64_12.dll）注入 PATH，
  使 onnxruntime GPU 版启用 CUDAExecutionProvider（YOLO ~7ms/帧）。
- 验证 EP：`onnxruntime.get_available_providers()` 应含 CUDAExecutionProvider。
- 若在无 GPU 机器运行：自动回退 CPUExecutionProvider，功能不变、速度较慢。

### 4.3 依赖说明（2026-08-27 起 third_party/ 已 gitignore）
- 日常推理用 ONNX 引擎，**不需要** `third_party/ultralytics`。
- 若需**重训**（train_yolov8.py）或**重新导出 ONNX**（export_onnx.py），需先恢复源码：
  ```powershell
  Copy-Item -Recurse D:\project\step1\week12\ultralytics-8.4.113\ultralytics third_party\
  ```
  （或 `pip install ultralytics==8.4.113`，但 train_yolov8.py 要求项目内源码 + 版本精确匹配。）
