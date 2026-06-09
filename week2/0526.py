import gradio as gr
import pandas as pd

# ================= 数据层 =================
cards = [
    {"name": "张三", "tel": "13812345678", "job": "CEO", "addr": "四川"},
    {"name": "李四", "tel": "13987654321", "job": "CTO", "addr": "北京"},
    {"name": "王五", "tel": "13700001111", "job": "CFO", "addr": "上海"},
]

def get_dataframe():
    """将列表转换为 DataFrame 供表格组件渲染"""
    return pd.DataFrame(cards) if cards else pd.DataFrame(columns=["name", "tel", "job", "addr"])

def find_card(name):
    for card in cards:
        if card["name"] == name:
            return card
    return None

# ================= 业务逻辑层 =================
def add_card(name, tel, job, addr):
    if not name.strip():
        return get_dataframe(), "❌ 姓名不能为空！", "", "", "", ""
    if find_card(name.strip()):
        return get_dataframe(), f"❌ 姓名 '{name}' 已存在！", "", "", "", ""
    
    cards.append({"name": name.strip(), "tel": tel.strip(), "job": job.strip(), "addr": addr.strip()})
    return get_dataframe(), "✅ 添加成功！", "", "", "", ""

def edit_card(name, tel, job, addr):
    card = find_card(name.strip())
    if not card:
        return get_dataframe(), f"❌ 未找到 '{name}'，请确认姓名或先选中表格行"
    
    card.update({"name": name.strip(), "tel": tel.strip(), "job": job.strip(), "addr": addr.strip()})
    return get_dataframe(), "✅ 修改成功！"

def delete_card(name):
    card = find_card(name.strip())
    if not card:
        return get_dataframe(), f"❌ 未找到 '{name}'，请先在表格中点击选中"
    
    cards.remove(card)
    return get_dataframe(), f"✅ 已删除 '{name}'"

def on_table_select(evt: gr.SelectData):
    """点击表格行时，自动回填到输入框"""
    row_idx = evt.index[0]
    if 0 <= row_idx < len(cards):
        card = cards[row_idx]
        return card["name"], card["tel"], card["job"], card["addr"]
    return "", "", "", ""

# ================= UI 层 (Gradio Blocks) =================
with gr.Blocks(title="名片管理系统", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📇 名片管理系统")
    
    # 状态提示框
    status_msg = gr.Textbox(label="操作状态", interactive=False, max_lines=1)
    
    # 输入表单
    with gr.Row():
        name_input = gr.Textbox(label="姓名", placeholder="必填项")
        tel_input = gr.Textbox(label="电话")
    with gr.Row():
        job_input = gr.Textbox(label="职位")
        addr_input = gr.Textbox(label="地址")
    
    # 操作按钮组
    with gr.Row():
        add_btn = gr.Button("➕ 添加", variant="primary")
        edit_btn = gr.Button("✏️ 修改", variant="secondary")
        delete_btn = gr.Button("🗑️ 删除", variant="stop")
        refresh_btn = gr.Button("🔄 刷新列表")
    
    # 数据展示表格
    table = gr.Dataframe(
        value=get_dataframe,
        headers=["姓名", "电话", "职位", "地址"],
        label="名片列表（💡 点击任意行可自动填充到上方输入框）",
        interactive=False,
        wrap=True
    )
    
    # ========== 事件绑定 ==========
    # 添加成功后自动清空输入框
    add_btn.click(add_card, 
                  inputs=[name_input, tel_input, job_input, addr_input],
                  outputs=[table, status_msg, name_input, tel_input, job_input, addr_input])
    
    edit_btn.click(edit_card,
                   inputs=[name_input, tel_input, job_input, addr_input],
                   outputs=[table, status_msg])
    
    delete_btn.click(delete_card,
                     inputs=[name_input],
                     outputs=[table, status_msg])
    
    refresh_btn.click(get_dataframe, outputs=[table])
    
    # ⭐ 核心交互：点击表格行回填数据
    table.select(on_table_select, outputs=[name_input, tel_input, job_input, addr_input])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=False)