"""和风天气每日预报工具：生成密钥、签发 JWT、发送真实 GET 请求。"""

import argparse
import json
import math
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import jwt
import requests
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


def generate_keys(directory):
    """公钥交给平台；私钥留在本机。避免覆盖已有密钥。"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    private_path = directory / "ed25519-private.pem"
    public_path = directory / "ed25519-public.pem"
    if private_path.exists() or public_path.exists():
        raise ValueError("目录内已有密钥文件，请继续使用原文件或指定其他目录。")
    private_key = Ed25519PrivateKey.generate()
    private_path.write_bytes(private_key.private_bytes(
        serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    ))
    public_path.write_bytes(private_key.public_key().public_bytes(
        serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo,
    ))
    print(f"已生成私钥：{private_path.resolve()}（仅留在本机）")
    print(f"已生成公钥：{public_path.resolve()}（将完整内容粘贴至平台）")


def load_config(filename):
    path = Path(filename).resolve()
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    fields = ("api_host", "developer_id", "project_id", "credential_id", "private_key_path")
    for field in fields:
        value = config.get(field)
        if not isinstance(value, str) or not value.strip() or "YOUR_" in value:
            raise ValueError(f"请在配置文件内填写真实的 {field}。")
        config[field] = value.strip()
    # 只接受平台专属 Host，防止误把凭据发送到其他网站。
    host = config["api_host"]
    parsed = urlsplit("https://" + host)
    if (parsed.hostname != host or not host.endswith(".qweatherapi.com")
            or parsed.path or parsed.query or parsed.fragment or parsed.port):
        raise ValueError("api_host 仅填写控制台分配的域名，不加 https:// 或路径。")
    key_path = Path(config["private_key_path"])
    if not key_path.is_absolute():
        key_path = path.parent / key_path
    config["private_key_path"] = key_path
    return config


def create_token(config):
    """按官方要求使用 Ed25519 私钥签发短期 JWT。"""
    now = int(time.time())
    payload = {
        "iss": config["developer_id"],
        "sub": config["project_id"],
        "iat": now - 30,
        "exp": now + 900,
    }
    private_key = config["private_key_path"].read_text(encoding="utf-8")
    return jwt.encode(payload, private_key, algorithm="EdDSA",
                      headers={"kid": config["credential_id"]})


def get_weather(config, latitude, longitude, days=3):
    """输入经纬度与天数，返回平台实际响应的 JSON 字典。"""
    if not math.isfinite(latitude) or not -90 <= latitude <= 90:
        raise ValueError("纬度必须在 -90 到 90 之间。")
    if not math.isfinite(longitude) or not -180 <= longitude <= 180:
        raise ValueError("经度必须在 -180 到 180 之间。")
    if isinstance(days, bool) or not isinstance(days, int) or not 1 <= days <= 10:
        raise ValueError("预报天数必须为 1 到 10 的整数。")

    # 路径参数：注意顺序是先纬度、后经度，精确到两位小数。
    url = (f"https://{config['api_host']}/weather/v1/daily/"
           f"{latitude:.2f}/{longitude:.2f}")
    # 请求头：Bearer 后有一个空格。JWT 不是项目 ID 或凭据 ID。
    headers = {"Authorization": f"Bearer {create_token(config)}"}
    # 查询参数：requests 负责将其拼接到 ? 后并完成编码。
    params = {"days": days, "localTime": "true", "lang": "zh"}
    response = requests.get(url, headers=headers, params=params,
                            timeout=(10, 30), allow_redirects=False)
    # requests 会自动处理 gzip 解压；无需手动解压。
    if not 200 <= response.status_code < 300:
        hints = {
            400: "检查经纬度、天数以及请求参数。",
            401: "检查 Host、三个 ID、公钥与私钥匹配关系及电脑时间。",
            403: "检查控制台的服务权限、凭据限制及账户状态。",
            404: "检查 API Host 和请求路径。",
            429: "请求过于频繁或达到限制，请查看控制台后再试。",
        }
        raise RuntimeError(f"天气 API 返回 HTTP {response.status_code}。"
                           + hints.get(response.status_code, "请查看官方错误码文档。"))
    data = response.json()
    if not isinstance(data, dict) or not isinstance(data.get("days"), list) or not data["days"]:
        raise RuntimeError("响应中缺少非空 days 列表，请核对使用的 API 版本。")
    return data


def show_forecast(data):
    for day in data["days"]:
        start = day.get("forecastStartTime", "未知")
        end = day.get("forecastEndTime", "未知")
        high = day["temperatureMax"]
        low = day["temperatureMin"]
        daytime = day.get("daytime", {}).get("condition", {}).get("text", "暂无数据")
        nighttime = day.get("nighttime", {}).get("condition", {}).get("text", "暂无数据")
        print(f"预报时段：{start} 至 {end}")
        print(f"  白天：{daytime}；夜间：{nighttime}")
        print(f"  最低温：{low['value']}{low['unit']}；最高温：{high['value']}{high['unit']}")
    print("数据来源：和风天气 QWeather")
    for attribution in data.get("metadata", {}).get("attributions", []):
        print(f"来源说明：{attribution}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    keys = commands.add_parser("keygen", help="在本机生成 Ed25519 公钥和私钥")
    keys.add_argument("--dir", default="keys")
    forecast = commands.add_parser("forecast", help="真实请求每日天气预报")
    forecast.add_argument("--config", default="config.json")
    forecast.add_argument("--lat", type=float, required=True, help="纬度")
    forecast.add_argument("--lon", type=float, required=True, help="经度")
    forecast.add_argument("--days", type=int, default=3)
    forecast.add_argument("--save", help="保存平台原始 JSON，供作业分析")
    args = parser.parse_args()
    try:
        if args.command == "keygen":
            generate_keys(args.dir)
        else:
            data = get_weather(load_config(args.config), args.lat, args.lon, args.days)
            if args.save:
                Path(args.save).write_text(json.dumps(data, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
                print(f"原始响应已保存：{Path(args.save).resolve()}")
            show_forecast(data)
    except requests.RequestException:
        print("请求失败：请检查网络、API Host 和连接超时；未获取到天气数据。", file=sys.stderr)
        return 1
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, jwt.PyJWTError) as exc:
        print(f"运行失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
