cards = [
    {"name": "张三", "tel": "13812345678", "job": "CEO", "addr": "四川"},
    {"name": "李四", "tel": "13987654321", "job": "CTO", "addr": "北京"},
    {"name": "王五", "tel": "13700001111", "job": "CFO", "addr": "上海"},
]

def add_card():
    card = {}
    card["name"] = input("姓名: ")
    card["tel"] = input("电话: ")
    card["job"] = input("职位: ")
    card["addr"] = input("地址: ")
    cards.append(card)

def show_cards():
    for card in cards:
        print(card)

def find_card(name):
    for card in cards:
        if card["name"] == name:
            return card
    return None

def edit_card():
    name = input("请输入要修改的姓名: ")
    card = find_card(name)
    if card:
        card["name"] = input("新姓名: ")
        card["tel"] = input("新电话: ")
        card["job"] = input("新职位: ")
        card["addr"] = input("新地址: ")
        print("修改成功")
    else:
        print("未找到该名片")

def delete_card():
    name = input("请输入要删除的姓名: ")
    for i, card in enumerate(cards):
        if card["name"] == name:
            cards.pop(i)
            print("删除成功")
            return
    print("未找到该名片")
    
def main():
    while True:
        print("\n===== 名片管理系统 =====")
        print("1. 添加名片")
        print("2. 显示所有名片")
        print("3. 修改名片")
        print("4. 删除名片")
        print("0. 退出系统")
        print("==========================")
        
        choice = input("请选择操作(0-4): ").strip()
        
        if choice == "1":
            add_card()
        elif choice == "2":
            show_cards()
        elif choice == "3":
            edit_card()
        elif choice == "4":
            delete_card()
        elif choice == "0":
            print("已退出系统，再见！")
            break
        else:
            print("无效输入，请重新选择")

# 程序入口
if __name__ == "__main__":
    main()