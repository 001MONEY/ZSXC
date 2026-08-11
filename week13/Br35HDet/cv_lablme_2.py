import json
import os
root_dir =  r"D:\project\step1\week13\Br35HDet\Br35HDet"
json_path = r"D:\project\step1\week13\Br35HDet\Br35HDet\annotations_all_new.json"
data = json.load(open(json_path,"r",encoding="utf-8"))
# ✅ 输出目录（可修改）
OUTPUT_DIR = r"D:\project\step1\week13\Br35HDet\Br35HDet\images"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
split_names = ["TRAIN","TEST","VAL"]
for split_name in split_names:
    imgs =os.path.join(root_dir,split_name)
    for name in os.listdir(imgs):
        img_path  = os.path.join(imgs,name)
        img_data = data[name]

        # 构建 labelme 标准结构
        labelme_data = {
            "version": "5.4.1",  # 常用 LabelMe 版本，可按需修改
            "flags": {},
            "shapes": [],
            "imagePath": name,
            "imageData": None,  # 表示不内嵌 base64 图像（推荐：节省空间，依赖外部图片）
            "imageHeight": -1,   # 占位，实际使用时建议从图像读取（见下方注释）
            "imageWidth": -1
        }

        # 转换 regions → shapes
        for region in img_data["regions"]:
            sa = region.get("shape_attributes", {})
            ra = region.get("region_attributes", {})

            shape_name = sa.get("name", "polygon")
            xs = sa.get("all_points_x", [])
            ys = sa.get("all_points_y", [])



            points = [[float(x), float(y)] for x, y in zip(xs, ys)]

            labelme_data["shapes"].append({
                "label": "tumor",  # ✅ 可按需改为具体类别名，如从 region_attributes 提取
                "points": points,
                "group_id": None,
                "shape_type": shape_name,
                "flags": {},
                "attributes": ra  # 直接映射 region_attributes
            })

        # ✅ 写入 JSON 文件（注意：文件名与图片同名 + .json）
        json_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(name)[0]}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(labelme_data, f, indent=2, ensure_ascii=False)
        print(f"✅ 已生成: {json_path}")

    print(f"\n🎉 全部完成！JSON 文件已保存至: {os.path.abspath(OUTPUT_DIR)}")