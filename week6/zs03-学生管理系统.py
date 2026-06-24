from mysql_db import MySqlHelper


class Student:
    def __init__(self, mysql_helper):
        self.helper = mysql_helper
        self.table = "student"  # 对应 db_0623_1 中的 student 表

    def student_menu(self):
        while True:
            print("-----------学生信息菜单----------")
            print("\n 1) 查询所有学生信息")
            print(" 2) 新增学生信息")
            print(" 3) 修改学生信息")
            print(" 4) 删除学生信息")
            print(" 5) 返回上一层菜单")
            no = input("请输入选择的序号：")

            if no == '1':
                self.show_students()
                input("按任意键返回上层菜单")
            elif no == '2':
                self.add_student()
                input("按任意键返回上层菜单")
            elif no == '3':
                self.edit_student()
                input("按任意键返回上层菜单")
            elif no == '4':
                self.delete_student()
                input("按任意键返回上层菜单")
            elif no == '5':
                break
            else:
                print("输入错误！请重新输入！")

    def show_students(self):
        """查询所有学生"""
        sql = "SELECT * FROM student"
        res = self.helper.find_all(sql)

        if not res:
            print("暂无学生数据")
            return

        print(f"\n{'='*65}")
        print(f"{'ID':<6} {'姓名':<10} {'性别':<6} {'出生':<8} {'院系':<16} {'地址':<20}")
        print(f"{'-'*65}")
        for stu in res:
            print(f"{stu['Id']:<6} {stu['Name']:<10} {stu['Sex']:<6} {stu['Birth']:<8} {stu['Department']:<16} {stu['Address']:<20}")
        print(f"{'='*65}")
        print(f"共 {len(res)} 条记录\n")

    def add_student(self):
        """新增学生"""
        print("\n--- 新增学生 ---")
        sid = int(input("学号(ID): "))
        name = input("姓名: ")
        sex = input("性别(男/女): ")
        birth = int(input("出生年份: "))
        dept = input("院系: ")
        addr = input("地址: ")

        sql = "INSERT INTO student (Id, Name, Sex, Birth, Department, Address) VALUES (%s, %s, %s, %s, %s, %s)"
        try:
            self.helper.execute(sql, (sid, name, sex, birth, dept, addr))
            print(f"✅ 新增成功: {sid} {name}")
        except Exception as e:
            print(f"❌ 新增失败: {e}")

    def edit_student(self):
        """修改学生"""
        print("\n--- 修改学生 ---")
        sid = int(input("请输入要修改的学号(ID): "))

        sql = "SELECT * FROM student WHERE Id = %s"
        stu = self.helper.find_one(sql, (sid,))
        if not stu:
            print(f"❌ 未找到学号 {sid}")
            return

        print(f"原信息: {stu}")
        name = input(f"姓名({stu['Name']}): ") or stu['Name']
        sex = input(f"性别({stu['Sex']}): ") or stu['Sex']
        birth_str = input(f"出生年份({stu['Birth']}): ")
        birth = int(birth_str) if birth_str else stu['Birth']
        dept = input(f"院系({stu['Department']}): ") or stu['Department']
        addr = input(f"地址({stu['Address']}): ") or stu['Address']

        sql = "UPDATE student SET Name=%s, Sex=%s, Birth=%s, Department=%s, Address=%s WHERE Id=%s"
        self.helper.execute(sql, (name, sex, birth, dept, addr, sid))
        print(f"✅ 修改成功: 学号 {sid}")

    def delete_student(self):
        """删除学生"""
        print("\n--- 删除学生 ---")
        sid = int(input("请输入要删除的学号(ID): "))

        sql = "SELECT * FROM student WHERE Id = %s"
        stu = self.helper.find_one(sql, (sid,))
        if not stu:
            print(f"❌ 未找到学号 {sid}")
            return

        print(f"即将删除: {stu}")
        confirm = input("确认删除？(y/n): ")
        if confirm.lower() == 'y':
            self.helper.execute("DELETE FROM student WHERE Id = %s", (sid,))
            print(f"✅ 已删除学号 {sid}")
        else:
            print("已取消")


class Course:
    """课程信息管理（对应 course_sys 表）"""

    def __init__(self, mysql_helper):
        self.helper = mysql_helper
        self.table = "course_sys"

    def course_menu(self):
        while True:
            print("-----------课程信息菜单----------")
            print("\n 1) 查看所有课程")
            print(" 2) 新增课程")
            print(" 3) 删除课程")
            print(" 4) 返回上一层菜单")
            no = input("请输入选择的序号：")

            if no == '1':
                self.show_courses()
                input("按任意键返回上层菜单")
            elif no == '2':
                self.add_course()
                input("按任意键返回上层菜单")
            elif no == '3':
                self.delete_course()
                input("按任意键返回上层菜单")
            elif no == '4':
                break
            else:
                print("输入错误！请重新输入！")

    def show_courses(self):
        sql = "SELECT * FROM course_sys"
        res = self.helper.find_all(sql)
        if not res:
            print("暂无课程数据")
            return
        print(f"\n{'='*50}")
        for c in res:
            print(f"  {c['cNo']}  {c['cName']}  (创建者: {c['createUser']})")
        print(f"{'='*50}")
        print(f"共 {len(res)} 门课程\n")

    def add_course(self):
        cno = input("课程编号: ")
        cname = input("课程名称: ")
        sql = "INSERT INTO course_sys (cNo, cName, createUser, createTime) VALUES (%s, %s, %s, NOW())"
        try:
            self.helper.execute(sql, (cno, cname, "admin"))
            print(f"✅ 新增课程成功: {cno} {cname}")
        except Exception as e:
            print(f"❌ 新增失败: {e}")

    def delete_course(self):
        cno = input("请输入要删除的课程编号: ")
        sql = "DELETE FROM course_sys WHERE cNo = %s"
        self.helper.execute(sql, (cno,))
        print(f"✅ 已删除课程 {cno}")


def main_menu():
    # 连接已存在的 db_0623_1 数据库
    my_sql = MySqlHelper(host="localhost", port=3306, user='root', pwd='8888', db_name='db_0623_1')
    if not my_sql.connect():
        print("数据库连接失败，程序退出")
        return

    print("---------欢迎使用学生管理系统v1.0--------------")
    print(f"数据库: {my_sql.db_name}")

    while True:
        print("\n 1) 学生信息")
        print(" 2) 课程信息")
        print(" 3) 退出\n")

        no = input("请输入选择的序号：")
        if no == '1':
            Student(my_sql).student_menu()
        elif no == '2':
            Course(my_sql).course_menu()
        elif no == '3':
            print("👋 再见！")
            break
        else:
            print("输入错误！请重新输入！")

    my_sql.close()


if __name__ == '__main__':
    main_menu()




