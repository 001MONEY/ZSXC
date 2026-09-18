"""
参数模块
"""
from pathlib import Path
path = Path(__file__).resolve().parent


class Config:
    FILE_PATH = path / "raw" / "cmn.txt"

