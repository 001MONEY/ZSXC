#Store


'''
样例1
'''
from langgraph.store.memory import InMemoryStore
import uuid

# 初始化内存 store
store = InMemoryStore()

# 写入数据
namespace = ("memories", "user_1")
store.put(namespace, str(uuid.uuid4()), {"value":"User name is Bob"})
store.put(namespace, str(uuid.uuid4()), {"value":"User likes Python"})



# 语义搜索
# Store 内部会对它做 embedding，然后跟 namespace 下所有 value 的向量做相似度匹配，按相关性排序返回。
# 返回值 memories：一个 Item 对象的列表，按相似度从高到低排序。
memories = store.search(namespace, query="Bob  likes python")
for d in memories:
    print(d.value["value"])

# 精确获取
# memory = store.get(namespace, key="某个uuid")