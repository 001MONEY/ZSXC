# # 测试 ord(字符) → 数字
# print("ord('a') =", ord('a'))
# print("ord('A') =", ord('A'))

# # 测试 chr(数字) → 字符（只用32-126的可见字符）
# print("chr(97) =", chr(97))
# print("chr(35) =", chr(35))  # #号
# print("chr(49) =", chr(49))  # 1


# num = int(input("每天销售多少碗："))
# price = float(input("一碗面的价格："))
# days = int(input("每年营业多久："))
# total   = num * price * days    #合计
# print(f'销售额是:{total}')



# 大写字母 A-Z
print("大写字母 A-Z:")
i = ord('A')
while i <= ord('Z'):
    print(f"{chr(i)} → {i}")
    i += 1

print("\n------------------------\n")

# 小写字母 a-z
print("小写字母 a-z:")
i = ord('a')
while i <= ord('z'):
    print(f"{chr(i)} → {i}")
    i += 1