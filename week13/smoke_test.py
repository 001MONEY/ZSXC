# ============================================================
# 冒烟测试：验证 imgsz=512, batch=16 在 6GB 显存上能否跑通
# 只训练 1 个 epoch，输出到独立目录 runs/smoke_test，不影响正式训练
# ============================================================
import sys
import torch
sys.path.insert(0, r"d:\project\step1\week12\ultralytics-8.4.113")

from ultralytics import YOLO


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"显存: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")

    model = YOLO(r"D:\project\step1\week13\yolov8s-seg.pt")

    try:
        model.train(
            data=r"D:\project\step1\week13\Br35HDet\yolodataset\dataset.yaml",
            epochs=1,          # 冒烟测试只跑 1 轮
            imgsz=512,
            batch=16,
            amp=False,
            project=r"D:\project\step1\week13\runs",
            name="smoke_test",
            exist_ok=True,
            verbose=True,
        )
        print("\n=== 冒烟测试通过 ✅ imgsz=512, batch=16 可正常训练 ===")
    except torch.cuda.OutOfMemoryError as e:
        print("\n=== 冒烟测试失败 ❌ 爆显存 ===")
        print(f"CUDA out of memory: {e}")
        sys.exit(2)


# Windows 下 DataLoader 子进程 spawn 需要该保护块
if __name__ == "__main__":
    main()
