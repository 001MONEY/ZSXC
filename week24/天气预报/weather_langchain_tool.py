"""可供 LangChain 模型调用的真实和风天气工具；与 weather_tool.py、config.json 放在同一目录。"""

import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Literal

import jwt
import requests
from langchain_core.tools import tool, ToolException
from pydantic import BaseModel, ConfigDict, Field


def find_weather_dir(start: Path) -> Path:
    """从 Notebook 当前目录向上查找已有的天气工具，不依赖原 E 盘路径。"""
    for base in [start.resolve(), *start.resolve().parents]:
        for candidate in (base / "天气预报", base / "week24" / "天气预报", base):
            if (candidate / "weather_tool.py").is_file() and (candidate / "config.json").is_file():
                return candidate
    raise FileNotFoundError("未找到天气预报目录；请将 Notebook 与天气预报文件夹放在同一课程目录")


WEATHER_DIR = Path(__file__).resolve().parent
if str(WEATHER_DIR) not in sys.path:
    sys.path.insert(0, str(WEATHER_DIR))
import weather_tool as qweather_api

# 使用已经验证过的示例坐标。城市名 -> 坐标属于本地映射，不是在线城市搜索。
# 当前凭据只启用天气预报，未调用 GeoAPI；暂只支持以下两个城市的示例位置。
WEATHER_LOCATIONS = {"成都": (30.67, 104.07), "北京": (39.92, 116.41)}


class RealWeatherArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    city: Literal["成都", "北京"] = Field(description="查询城市，目前支持成都或北京的示例位置")
    days: int = Field(default=3, ge=1, le=10, strict=True, description="未来预报天数，1～10，默认3天")


@tool(args_schema=RealWeatherArgs)
def query_weather_forecast(city: Literal["成都", "北京"], days: int = 3) -> str:
    """通过和风天气 API 查询成都或北京未来1～10天的真实每日天气预报。

    返回预报时段、白天/夜间天气和最高/最低温及单位。这是每日预报，不是当前实况。
    只支持成都或北京的示例坐标；缺少城市先追问，其他城市先说明目前不支持。
    """
    latitude, longitude = WEATHER_LOCATIONS[city]
    try:
        # config.json 与私钥由程序读取，不属于模型的输入 schema。
        local_config = qweather_api.load_config(WEATHER_DIR / "config.json")
        raw = qweather_api.get_weather(local_config, latitude, longitude, days)
    except requests.RequestException as exc:
        raise ToolException("天气服务连接失败或超时，请稍后重试；未获得本次预报。") from exc
    except OSError as exc:
        raise ToolException("本地天气配置或私钥不可读，请检查天气预报目录。") from exc
    except jwt.PyJWTError as exc:
        raise ToolException("天气服务 JWT 生成失败，请检查本地认证配置。") from exc
    except (ValueError, RuntimeError) as exc:
        raise ToolException(f"天气查询失败：{exc}") from exc

    # 精简返回值，保留时间、单位与来源，避免将整份庞大 JSON 发给模型。
    try:
        forecast_days = [
            {
                "forecastStartTime": day["forecastStartTime"],
                "forecastEndTime": day["forecastEndTime"],
                "daytime": day.get("daytime", {}).get("condition", {}).get("text"),
                "nighttime": day.get("nighttime", {}).get("condition", {}).get("text"),
                "temperatureMin": day["temperatureMin"],
                "temperatureMax": day["temperatureMax"],
            }
            for day in raw["days"]
        ]
    except (KeyError, TypeError) as exc:
        raise ToolException("天气服务返回的数据结构与每日预报接口不符。") from exc
    return json.dumps(
        {
            "ok": True,
            "source": "QWeather",
            "data_type": "daily_forecast",
            "city": city,
            "location": {"latitude": latitude, "longitude": longitude, "basis": "本地示例坐标"},
            "queried_at": datetime.now(timezone.utc).isoformat(),
            "days": forecast_days,
            "attributions": raw.get("metadata", {}).get("attributions", []),
        },
        ensure_ascii=False,
    )
