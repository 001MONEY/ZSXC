"""
================================================================================
本地测试客户端：向 Docker 里的 YOLO 推理服务发请求
================================================================================
用法（在 week15/yolo_onnx_service 目录下）：
    D:/project/step1/env/python.exe client_test.py ../../week13/img.jpg

会依次做三件事：
    1. GET  /health   健康检查
    2. POST /detect   上传图片 → 打印检测结果
    3. POST /annotate 上传图片 → 保存画好框的标注图
================================================================================
"""

import sys

import httpx

BASE_URL = "http://127.0.0.1:8001"   # 与 docker-compose.yml 里的端口一致


def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else "../../week13/img.jpg"

    # 1) 健康检查
    r = httpx.get(f"{BASE_URL}/health")
    print("【health】", r.json())
    print("-" * 60)

    # 2) 目标检测
    with open(img_path, "rb") as f:
        r = httpx.post(f"{BASE_URL}/detect",
                       files={"file": (img_path, f, "image/jpeg")})
    print("【detect】状态码：", r.status_code)
    data = r.json()
    print(f"图片尺寸：{data['width']}x{data['height']}，检测到 {data['num_detections']} 个目标")
    for d in data["detections"]:
        print(f"  {d['class_name']:<15} 置信度={d['confidence']:<6} bbox={d['bbox']}")
    print("-" * 60)

    # 3) 画框标注图
    with open(img_path, "rb") as f:
        r = httpx.post(f"{BASE_URL}/annotate",
                       files={"file": (img_path, f, "image/jpeg")})
    out = "annotated_out.jpg"
    with open(out, "wb") as f:
        f.write(r.content)
    print("【annotate】标注图已保存：", out)


if __name__ == "__main__":
    main()
