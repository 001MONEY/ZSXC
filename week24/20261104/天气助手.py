import requests

from langchain.tools import tool

from pydantic import BaseModel,Field


class City(BaseModel):
    city: str = Field(description="城市的地理位置")

class LocalId(BaseModel):
    city: str = Field(description="城市的地理位置")

@tool(args_schema=City)
def get_local_id(city:str) -> str:
    """获取城市的地理位置id"""

    url = "https://ky46h2txb7.re.qweatherapi.com/geo/v2/city/lookup"
    response = requests.get(url=url,params={"location":city,"key":"0b20fda53afc4b04a075941f6857f489"})

    if response.status_code == 200:
        data = response.json()
        if data["code"] == "200":
            location_id = data["location"][0]["id"]
            return location_id
        else:
            return f"Error: {data['code']}"
    else:
        return f"Error: {response.status_code}"


class LocalId(BaseModel):
    city: str = Field(description="城市的地理位置")

@tool(args_schema=LocalId)
def get_weather(location_id:str) -> str:

    url = 'https://ky46h2txb7.re.qweatherapi.com/v7/weather/now'

    response = requests.get(url=url,params={"location":location_id,"key":"0b20fda53afc4b04a075941f6857f489"})
    if response.status_code == 200:
        data = response.json()
        if data["code"] == "200":
            weather = data["now"]
            return weather
        else:
            return f"Error: {data['code']}"
    else:
        return f"Error: {response.status_code}"
        



