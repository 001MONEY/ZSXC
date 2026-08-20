# -*- coding: utf-8 -*-
"""TensorRT 引擎加载工具。

兼容 ultralytics 导出的 .engine 文件格式：
    4字节小端长度 + JSON 元数据 + 引擎二进制
也兼容纯 TRT plan 文件（无元数据头）。
"""
import json
import os
import tensorrt as trt

_TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def ensure_trt_path():
    """把手动安装的 TensorRT lib 加入 DLL 搜索路径（须在 import tensorrt 前调用）。"""
    trt_lib = r"D:\TensorRT-10.10.0.31\TensorRT-10.10.0.31\lib"
    if os.path.isdir(trt_lib) and trt_lib not in os.environ.get("PATH", ""):
        os.environ["PATH"] = trt_lib + os.pathsep + os.environ.get("PATH", "")


def strip_metadata(data: bytes) -> bytes:
    """剥离 ultralytics 引擎文件开头的元数据头，返回纯引擎二进制。"""
    if len(data) >= 4:
        meta_len = int.from_bytes(data[:4], "little")
        # 前 4 字节长度合理，且其后能解析为 JSON → 判定带元数据头
        if 0 < meta_len < len(data) - 4:
            try:
                json.loads(data[4:4 + meta_len].decode("utf-8"))
                return data[4 + meta_len:]
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return data  # 没有元数据头，直接返回


def load_engine(path: str) -> trt.ICudaEngine:
    """从 .engine 文件加载 TensorRT 引擎（自动处理元数据头）。"""
    ensure_trt_path()
    with open(path, "rb") as f:
        data = f.read()
    engine_bytes = strip_metadata(data)
    runtime = trt.Runtime(_TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_bytes)
    if engine is None:
        raise RuntimeError(f"引擎反序列化失败: {path}")
    return engine


def print_engine_io(engine: trt.ICudaEngine):
    """打印引擎输入/输出张量信息。"""
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        print(f"  {name:10s} shape={engine.get_tensor_shape(name)}  "
              f"{engine.get_tensor_dtype(name)}  {engine.get_tensor_mode(name)}")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        print(f"加载引擎: {p}")
        e = load_engine(p)
        print_engine_io(e)
