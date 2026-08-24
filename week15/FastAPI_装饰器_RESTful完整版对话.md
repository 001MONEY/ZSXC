### 用户：在用装饰器的时候，我要不要单独写一个py文件来定义各种装饰器，然后在其他py文件里面import

## 助手：

是的，在工程项目里通常会单独写一个 `decorators.py`（或者
`utils/decorators.py`）来集中管理装饰器，然后其他模块通过 `import`
使用。这是比较标准的做法。

例如项目结构：

```text
project/
│
├── main.py
├── model.py
├── train.py
│
├── utils/
│   ├── __init__.py
│   └── decorators.py
│
└── config.py
```

`decorators.py`：

```python
import time
import functools


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()

        result = func(*args, **kwargs)

        end = time.time()
        print(f"{func.__name__}运行时间:{end-start:.4f}s")

        return result

    return wrapper
```

然后其他文件：

```python
from utils.decorators import timer


@timer
def train_model():
    print("训练模型")


train_model()
```

输出：

```text
训练模型
train_model运行时间:0.0021s
```

------------------------------------------------------------------------

### 为什么工程里面要单独管理装饰器？

小项目：

```python
train.py
```

里面直接写装饰器没有问题。

但是项目变大：

```text
train.py
infer.py
evaluate.py
api.py
database.py
```

都会需要：

- 时间统计
- 日志
- 权限验证
- 异常处理
- 自动重试

如果每个文件复制一份，会造成维护困难。

所以工程里面一般：

```text
utils/
    decorators.py
```

统一管理。

------------------------------------------------------------------------

### AI工程中的装饰器应用

例如：

#### 推理耗时统计

```python
@timeit
def inference(image):
    result=model(image)
```

输出：

```text
YOLO推理耗时:12ms
```

#### 异常捕获

```python
@exception_handler
def predict():
    ...
```

#### 日志记录

```python
@logger
def train():
    ...
```

------------------------------------------------------------------------

## 用户：今天要讲fast api

## 助手：

FastAPI 对 AI 算法工程方向很关键，它位于模型训练和实际部署之间。

可以理解为：

> FastAPI = 用 Python 快速搭建一个后端接口，让别人通过 HTTP
> 调用你的模型。

典型流程：

```text
摄像头
 ↓
上传图片
 ↓
FastAPI服务器
 ↓
YOLO模型推理
 ↓
返回结果
```

------------------------------------------------------------------------

## 最简单的 FastAPI

安装：

```bash
pip install fastapi uvicorn
```

代码：

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def hello():
    return {
        "message": "Hello FastAPI"
    }
```

启动：

```bash
uvicorn main:app --reload
```

------------------------------------------------------------------------

## 路由

例如：

```python
@app.get("/predict")
def predict():
    return {
        "class": "apple",
        "confidence": 0.98
    }
```

访问：

```text
http://127.0.0.1:8000/predict
```

返回 JSON。

------------------------------------------------------------------------

## 上传图片接口

```python
from fastapi import UploadFile, File


@app.post("/detect")
async def detect(image: UploadFile = File(...)):

    content = await image.read()

    return {
        "filename": image.filename,
        "size": len(content)
    }
```

------------------------------------------------------------------------

## 接入 YOLO

结构：

```text
project

├── main.py
├── weights
│    └── best.pt
└── model.py
```

模型文件负责推理。

FastAPI负责接口。

------------------------------------------------------------------------

## 用户：封装是在onnx文件的基础上做还是pth文件做还是engine文件

## 助手：

FastAPI封装可以基于 `.pth`、`.onnx`、`.engine`。

但是工程中更常见的是最终部署格式。

------------------------------------------------------------------------

## 基于 pth

流程：

```text
图片
 ↓
FastAPI
 ↓
PyTorch模型(pth)
 ↓
GPU推理
 ↓
返回结果
```

优点：

- 开发简单
- 调试方便

缺点：

- 依赖PyTorch
- 部署环境重

------------------------------------------------------------------------

## 基于 ONNX

流程：

```text
pth
 ↓
ONNX
 ↓
ONNX Runtime
 ↓
FastAPI
```

优点：

- 跨平台
- 工业部署常见

------------------------------------------------------------------------

## 基于 TensorRT engine

流程：

```text
PyTorch
 ↓
pth
 ↓
ONNX
 ↓
TensorRT
 ↓
engine
```

FastAPI：

```text
FastAPI

 ↓

TensorRT Runtime

 ↓

CUDA

 ↓

GPU
```

这是 NVIDIA GPU 高性能部署方案。

------------------------------------------------------------------------

## 企业实际流程：

```text
训练模型
 |
 ↓
best.pth
 |
 ↓
ONNX
 |
 ↓
TensorRT engine
 |
 ↓
FastAPI
 |
 ↓
Docker部署
```

------------------------------------------------------------------------

## 用户上传FastAPI课程目录图片

课程结构：

```text
装饰器
 ↓
HTTP
 ↓
RESTful
 ↓
FastAPI
```

------------------------------------------------------------------------

## 用户理解：

装饰器已经学习过。

HTTP协议在计算机网络课程中学过。

RESTful第一次接触。

------------------------------------------------------------------------

## 助手：

这个课程设计实际上是在补齐后端接口开发链路：

```text
装饰器
 ↓
HTTP协议
 ↓
RESTful API设计
 ↓
FastAPI框架
 ↓
模型服务部署
```

------------------------------------------------------------------------

# RESTful

RESTful 是一种 API 设计规范。

核心思想：

> 使用 HTTP 方法表示对资源的操作。

------------------------------------------------------------------------

## API是什么

API：

Application Programming Interface。

简单理解：

> 一个程序给另一个程序提供功能的入口。

例如：

Python内部：

```python
result=model(image)
```

如果网页或者APP想调用：

设计接口：

```text
POST /detect
```

上传图片：

```json
{
    "image":"xxx.jpg"
}
```

返回：

```json
{
    "class":"cat",
    "confidence":0.98
}
```

这就是API。

------------------------------------------------------------------------

# RESTful规则

常见HTTP方法：

  方法     作用

  -------- ----------

  GET      获取数据
  POST     提交数据
  PUT      修改数据
  DELETE   删除数据

------------------------------------------------------------------------

例如：

查询模型：

```http
GET /models
```

上传图片检测：

```http
POST /detect
```

删除模型：

```http
DELETE /models/yolo
```

------------------------------------------------------------------------

# RESTful在AI项目中的应用

智能称重项目：

```text
摄像头
 ↓
FastAPI
 ↓
TensorRT engine
 ↓
YOLO
 ↓
返回商品、重量、价格
```

接口：

```http
POST /detect
```

返回：

```json
{
"product":"apple",
"weight":523,
"price":5.2
}
```

------------------------------------------------------------------------

# 总结

对于 AI 算法工程方向：

需要掌握：

1. 装饰器
2. HTTP基础
3. RESTful接口设计
4. FastAPI路由
5. 参数处理
6. 文件上传
7. 模型推理接口封装
8. Docker部署

完整链路：

```text
模型训练
 ↓
pth
 ↓
ONNX
 ↓
TensorRT engine
 ↓
FastAPI
 ↓
Docker
 ↓
工业应用
```
