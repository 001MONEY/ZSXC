#测试连接

from pymilvus import MilvusClient

client = MilvusClient(uri="http://192.168.2.197:19530")

print("连接成功！")
print("Milvus 版本：", client.get_server_version())

client.close()  # 断开连接