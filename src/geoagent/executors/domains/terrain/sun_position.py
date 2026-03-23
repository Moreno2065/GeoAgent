"""
SunPosition - 太阳位置计算模块
============================
用于计算任意时间任意地点的太阳方位角和高度角。

用于：
  - 阴影分析
  - 太阳辐射分析
  - 太阳能潜力评估
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Tuple


def calculate_sun_position(
    dt: datetime,
    latitude: float,
    longitude: float = 0.0
) -> Tuple[float, float]:
    """
    计算太阳位置（方位角和高度角）

    基于 NOAA Solar Calculator 算法。

    Args:
        dt: 日期时间对象
        latitude: 纬度（度）
        longitude: 经度（度）

    Returns:
        (方位角, 高度角) 单位：度
        - 方位角：正北为0度，顺时针增加
        - 高度角：地平线为0度，正上方为90度

    示例:
        >>> from datetime import datetime
        >>> dt = datetime(2024, 6, 21, 12, 0)
        >>> azimuth, altitude = calculate_sun_position(dt, 39.9, 116.4)
        >>> print(f"方位角: {azimuth:.2f}°, 高度角: {altitude:.2f}°")
    """
    year = dt.year
    month = dt.month
    day = dt.day
    hour = dt.hour + dt.minute / 60 + dt.second / 3600

    # 儒略日计算
    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + A // 4
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + hour / 24 + B - 1524.5

    # 儒略世纪数
    T = (JD - 2451545.0) / 36525.0

    # 太阳几何参数
    # 太阳平均黄经
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360

    # 太阳平近点角
    M = (357.52911 + 35999.0503 * T - 0.0001537 * T * T) % 360
    M_rad = math.radians(M)

    # 太阳中心方程
    C = (1.9146 - 0.004817 * T - 0.000014 * T * T) * math.sin(M_rad) + \
        (0.019993 - 0.000101 * T) * math.sin(2 * M_rad) + \
        0.00029 * math.sin(3 * M_rad)

    # 太阳真实黄经
    lambda_sun = L0 + C

    # 太阳黄纬（近似为0）
    beta = 0

    # 太阳赤纬
    epsilon = 23.439291 - 0.013004 * T - 0.00000016 * T * T + 0.000000504 * T * T * T
    epsilon_rad = math.radians(epsilon)
    lambda_rad = math.radians(lambda_sun)
    beta_rad = math.radians(beta)

    delta = math.asin(math.sin(beta_rad) * math.cos(epsilon_rad) +
                     math.cos(beta_rad) * math.sin(epsilon_rad) * math.sin(lambda_rad))

    # 计算地方时角
    # 参考太阳时
    J0 = 0.0009 + 1.0027379093 * (hour - 12)  # 简化：假设正午为12
    theta_0 = 280.46061837 + 360.98564736629 * (JD - 2451545.0) + \
              0.000387933 * T * T - T * T * T / 38710000.0
    theta_0 = theta_0 % 360

    # 地方恒星时
    LST = (theta_0 + longitude) % 360

    # 太阳时角
    H = LST - lambda_sun
    H_rad = math.radians(H)

    lat_rad = math.radians(latitude)

    # 太阳高度角
    sin_alt = (math.sin(lat_rad) * math.sin(delta) +
               math.cos(lat_rad) * math.cos(delta) * math.cos(H_rad))
    sin_alt = max(-1.0, min(1.0, sin_alt))
    altitude = math.degrees(math.asin(sin_alt))

    # 太阳方位角
    cos_az = (math.sin(delta) - math.sin(lat_rad) * sin_alt) / \
             (math.cos(lat_rad) * math.cos(math.asin(sin_alt)))
    cos_az = max(-1.0, min(1.0, cos_az))

    if math.sin(H_rad) > 0:
        azimuth = 360.0 - math.degrees(math.acos(cos_az))
    else:
        azimuth = math.degrees(math.acos(cos_az))

    return azimuth, altitude


def calculate_day_length(
    date: datetime,
    latitude: float,
    longitude: float = 0.0
) -> Tuple[float, float, float]:
    """
    计算日长、日出和日落时间

    Args:
        date: 日期（时间部分被忽略）
        latitude: 纬度
        longitude: 经度

    Returns:
        (日长小时数, 日出时间, 日落时间)

    示例:
        >>> import datetime
        >>> date = datetime.datetime(2024, 6, 21)
        >>> day_length, sunrise, sunset = calculate_day_length(date, 39.9)
        >>> print(f"日长: {day_length:.2f}小时")
    """
    # 创建正午时间
    noon = date.replace(hour=12, minute=0, second=0, microsecond=0)

    azimuth, altitude = calculate_sun_position(noon, latitude, longitude)

    # 太阳高度角为0时的时角
    lat_rad = math.radians(latitude)

    # 使用近似公式
    # cos(时角) = -tan(纬度) * tan(赤纬)
    # 这里使用正午的高度角来估算

    if altitude > 0:
        # 估算日出日落时间
        # 每度时角 = 4分钟
        hour_angle = 90 - altitude  # 简化估算
        day_length_hours = hour_angle / 15.0 * 2 if hour_angle > 0 else 0

        # 估算日出日落
        noon_hour = 12.0
        delta_hour = (90 - altitude) / 15.0

        sunrise_hour = noon_hour - delta_hour
        sunset_hour = noon_hour + delta_hour
    else:
        day_length_hours = 0
        sunrise_hour = 0
        sunset_hour = 0

    return day_length_hours, sunrise_hour, sunset_hour


def calculate_solar_radiation(
    date: datetime,
    latitude: float,
    longitude: float = 0.0,
    hour: float = None
) -> float:
    """
    估算太阳辐射强度

    Args:
        date: 日期
        latitude: 纬度
        longitude: 经度
        hour: 小时（可选，默认12点）

    Returns:
        太阳辐射相对强度 (0-1)
    """
    if hour is None:
        hour = date.hour + date.minute / 60.0

    dt = date.replace(hour=int(hour), minute=int((hour % 1) * 60))

    azimuth, altitude = calculate_sun_position(dt, latitude, longitude)

    if altitude <= 0:
        return 0.0

    # 大气透射率简化模型
    # 考虑大气质量和天顶角
    zenith = 90 - altitude
    zenith_rad = math.radians(zenith)

    # 大气质量（简化）
    air_mass = 1.0 / math.cos(zenith_rad) if zenith < 85 else 10

    # 透射率（简化）
    transmission = 0.7 ** air_mass

    return max(0.0, transmission * math.sin(math.radians(altitude)))


def get_season_info(date: datetime, hemisphere: str = "north") -> dict:
    """
    获取季节信息

    Args:
        date: 日期
        hemisphere: 半球 ("north" 或 "south")

    Returns:
        季节信息字典
    """
    month = date.month
    day = date.day

    # 北半球季节定义
    if hemisphere == "north":
        seasons = {
            "spring": (3, 20),
            "summer": (6, 21),
            "autumn": (9, 23),
            "winter": (12, 22),
        }
    else:
        seasons = {
            "spring": (9, 23),
            "summer": (12, 22),
            "autumn": (3, 20),
            "winter": (6, 21),
        }

    current_season = None
    for season, (m, d) in seasons.items():
        if (month > m) or (month == m and day >= d):
            current_season = season

    if current_season is None:
        current_season = "winter"

    return {
        "season": current_season,
        "hemisphere": hemisphere,
        "approx_daylight_hours": {
            "spring": 12,
            "summer": 14,
            "autumn": 12,
            "winter": 10,
        }[current_season]
    }


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "calculate_sun_position",
    "calculate_day_length",
    "calculate_solar_radiation",
    "get_season_info",
]
