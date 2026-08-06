# -*- coding: utf-8 -*-
"""
骨龄评估系统 Gradio Demo
上传左手腕 X 光片 → 输出 13 骨标注图 + 等级得分明细 + 骨龄

生产配置: 全 ordinal 分类 + 数据驱动校准（GradientBoosting）

运行:
    python gradio_app.py
    # 浏览器打开 http://127.0.0.1:7860
"""
import cv2
import gradio as gr
import numpy as np
import pandas as pd

from pipeline import Pipeline

# 生产配置：全部关节 ordinal + 校准模型（模型预加载）
PIPE = Pipeline(ordinal_all=True, calibrated=True)


def run_pipeline(image_path, sex):
    """上传图 -> 标注图(RGB), 明细表, 骨龄, 检出信息"""
    res = PIPE.predict(image_path, sex=sex)

    # 可视化标注图（BGR -> RGB）
    vis = PIPE.visualize(res)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    # 13 骨等级得分明细表
    rows = [{"骨头": d["bone"],
             "等级": d["grade"] if d["grade"] is not None else "-",
             "RUS得分": d["score"] if d["score"] is not None else "-"}
            for d in res["detail"]]
    df = pd.DataFrame(rows)

    # 骨龄文本
    age_txt = f"{res['bone_age_years']:.2f} 岁"
    if res.get("bone_age_months"):
        age_txt += f"（{res['bone_age_months']:.0f} 个月）"
    age_txt += f"  |  {sex}"

    missing = res.get("missing", [])
    info = f"检出 {res['n_bones']}/13 骨；缺失: {', '.join(missing) if missing else '无'}"
    return vis_rgb, df, age_txt, info


CSS = """
footer {display:none !important}
.gradio-container {max-width: 1100px !important; margin: auto !important}
h1 {text-align:center}
"""

with gr.Blocks(title="骨龄评估系统 Demo") as demo:
    gr.Markdown("# 骨龄评估系统 Demo")
    gr.Markdown("**两阶段方案**：YOLOv8 检测 → 13 骨过滤 → 9 关节 ordinal 分类 → 数据驱动校准回归骨龄")
    with gr.Row():
        with gr.Column(scale=1):
            img_in = gr.Image(type="filepath", label="上传左手腕正面 X 光片")
            sex = gr.Radio(["boy", "girl"], value="boy", label="性别")
            btn = gr.Button("评估骨龄", variant="primary")
        with gr.Column(scale=1):
            img_out = gr.Image(label="检测结果（13 骨框 + 等级）")
            age = gr.Label(label="骨龄")
            info = gr.Textbox(label="检出信息", interactive=False)
    out_df = gr.Dataframe(label="13 骨等级与 RUS 得分明细", interactive=False,
                          headers=["骨头", "等级", "RUS得分"])

    btn.click(run_pipeline, [img_in, sex], [img_out, out_df, age, info])
    img_in.change(run_pipeline, [img_in, sex], [img_out, out_df, age, info],
                  show_progress="hidden")


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, css=CSS)
