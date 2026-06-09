print(3,4,3+4)
age=888666
print(age)

name = input("Enter your name: ")
pwd = input("enter your passwod:")
print(name,pwd)

#单行注释
print(1,end='')   
print(2,3)


#多行注释
# name = 'lihua'
# print(name)
# address = id(name)
# print(address)
# print(id('lihua'))

uid = input('输入账号')
pwd = input('输入密码')
print('账号：',uid,'密码',pwd)

num     = 100                   #销量
price   = 15                    #单价
days    = 365                   #时间
total   = num * price * days    #合计
print(f'销售额是:{total}')

age     = 10
print(type(age))
price   = 5.8
print(type(price))

print(bool(0))
print(bool(1))
print(bool(''))
print(bool(None))

print(int(10.8))
print(type(12.7))
print(float('12.7'))

print(str('12.7'))
print(str(12.7))
print(str(False))

print('aa'+'bb')
print(10+True)
print('a'*10)


user_input = input("请输入任意数据：")
# 打印输入的数据和它的类型
print(f"你输入的数据是：{user_input}")
print(f"该数据的类型是：{type(user_input)}")

name = input("请输入你的姓名：")
gender = input("请输入你的性别：")
address = input("请输入你的家庭地址：")

# 使用f-string格式化输出
print(f"{name},{gender},地址:{address}")


a = 1
b = 2
print(f"交换前:a = {a}, b = {b}")
a, b = b, a
print(f"交换后:a = {a}, b = {b}")

num1 = float(input("请输入第一个数字："))
num2 = float(input("请输入第二个数字："))  
#计算和
sum_result = num1 + num2  
#输出结果
print(f"两个数字的和是：{sum_result}")