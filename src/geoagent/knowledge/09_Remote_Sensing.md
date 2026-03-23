# 遥感与空间分析知识库

## 1. 遥感基础知识

### 1.1 电磁波谱与遥感

遥感（Remote Sensing）是通过传感器从远处获取目标信息的技术。

**电磁波谱分区**：
- 可见光：380-750nm（蓝、绿、红）
- 近红外：750-1400nm
- 短波红外：1400-3000nm
- 热红外：8-14μm
- 微波：1mm-1m

### 1.2 主要卫星与传感器

| 卫星 | 传感器 | 空间分辨率 | 重访周期 |
|------|--------|-----------|---------|
| Sentinel-2 | MSI | 10/20/60m | 5天 |
| Landsat-8/9 | OLI/TIRS | 30m | 16天 |
| MODIS | MODIS | 250/500/1000m | 1-2天 |
| GF-2 | PMS | 1m | 5天 |
| PlanetScope | PS2 | 3m | 每日 |

---

## 2. 遥感指数体系

### 2.1 植被指数

**NDVI（归一化植被指数）**
```python
NDVI = (NIR - Red) / (NIR + Red)
```
- 范围：-1 到 +1
- 用途：植被覆盖度、生长状态
- 阈值：
  - NDVI < 0：非植被（水体、冰雪）
  - 0-0.2：裸土
  - 0.2-0.5：稀疏植被
  - 0.5-0.8：中等密度植被
  - > 0.8：茂密植被

**EVI（增强植被指数）**
```python
EVI = G * (NIR - Red) / (NIR + C1 * Red - C2 * Blue + L)
# G=2.5, C1=6, C2=7.5, L=1
```
- 适用于高植被密度区域
- 减少大气和土壤背景影响

**SAVI（土壤调节植被指数）**
```python
SAVI = ((NIR - Red) / (NIR + Red + L)) * (1 + L)
# L=0.5（适用于稀疏植被区）
```
- 适用于半干旱地区

**MSAVI（修正土壤调节植被指数）**
```python
MSAVI = (2 * NIR + 1 - sqrt((2 * NIR + 1)² - 8 * (NIR - Red))) / 2
```

### 2.2 水体指数

**NDWI（归一化水体指数）**
```python
NDWI = (Green - NIR) / (Green + NIR)
```
- 用于普通水体提取
- NDWI > 0 通常表示水体

**MNDWI（改进归一化水体指数）**
```python
MNDWI = (Green - SWIR) / (Green + SWIR)
```
- 更好地分离城镇水体和植被

**LSWI（土地表面水分指数）**
```python
LSWI = (NIR - SWIR) / (NIR + SWIR)
```

### 2.3 建筑/裸土指数

**NDBI（归一化建筑指数）**
```python
NDBI = (SWIR - NIR) / (SWIR + NIR)
```
- 用于城市建成区提取

**NDBaI（归一化裸土指数）**
```python
NDBaI = (SWIR1 - SWIR2) / (SWIR1 + SWIR2)
```

---

## 3. Sentinel-2 波段详解

| 波段 | 中心波长(nm) | 空间分辨率 | 主要用途 |
|------|-------------|-----------|---------|
| B1 | 443 | 60m | 大气气溶胶 |
| B2 | 490 | 10m | 蓝光 |
| B3 | 560 | 10m | 绿光 |
| B4 | 665 | 10m | 红光 |
| B5 | 705 | 20m | 红边1 |
| B6 | 740 | 20m | 红边2 |
| B7 | 783 | 20m | 红边3 |
| B8 | 842 | 10m | 近红外(NIR) |
| B8A | 865 | 20m | NIR窄 |
| B9 | 945 | 60m | 水汽 |
| B10 | 1375 | 60m | 卷云 |
| B11 | 1610 | 20m | SWIR1 |
| B12 | 2190 | 20m | SWIR2 |

---

## 4. 遥感处理流程

### 4.1 预处理流程

```
原始影像 → 辐射校正 → 大气校正 → 几何校正 → 正射校正 → 产品
```

**辐射校正**：
- 辐射定标：将DN值转换为表观反射率
- 太阳高度角校正
- 气溶胶光学厚度校正

**大气校正**：
- 基于辐射传输模型（6S、MODTRAN）
- 经验方法（黑暗像元法）

### 4.2 云检测方法

**基于阈值**：
```python
# 简单蓝光阈值法
is_cloud = blue_band > threshold
```

**基于云指数**：
```python
CloudIndex = (Red - SWIR) / (Red + SWIR)
```

**QA波段掩膜**：
```python
# Sentinel-2 使用 SCL 波段
# 0: 无数据, 1: 晕斑, 3: 云影, 6: 水体
# 8: 晕云, 9: 卷云, 10: 植被, 11: 裸土
```

---

## 5. 变化检测方法

### 5.1 分类后比较法

1. 分别对两个时期影像分类
2. 逐像元比较分类结果
3. 生成变化矩阵

### 5.2 影像差值法

```python
# 简单差值
change = image2 - image1

# 比值法
ratio = image2 / (image1 + epsilon)

# 主成分差异法
# 对两个时期做PCA，取差异分量
```

### 5.3 变化向量分析(CVA)

```python
# 计算变化向量的幅度和方向
magnitude = sqrt(change_band1² + change_band2² + ...)
direction = arctan(change_band2 / change_band1)
```

---

## 6. 影像分类方法

### 6.1 非监督分类

**K-Means**：
1. 随机选择K个初始聚类中心
2. 计算每个像元到聚类中心的距离
3. 将像元分配到最近的聚类
4. 更新聚类中心
5. 重复直到收敛

```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(pixels)
```

### 6.2 监督分类

**最大似然分类**：
```python
# 基于贝叶斯决策
P(class|x) ∝ P(x|class) * P(class)
```

**支持向量机(SVM)**：
```python
from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(training_pixels, training_labels)
```

---

## 7. 常见应用场景

### 7.1 土地利用/覆盖分类

常用指数组合：
- NDVI：区分植被类型
- NDWI：识别水体
- NDBI：区分建筑和裸土

### 7.2 水体提取

推荐使用MNDWI + NDWI组合：
```python
# MNDWI > 0 且 NDWI > 0 且 NDVI < 0.3
```

### 7.3 城市热岛效应

```python
# 热岛强度 = 地表温度 - 区域平均温度
```

### 7.4 植被变化监测

时序NDVI分析：
- 趋势分析：线性回归
- 季节性分析：FFT
- 异常检测：阈值法

---

## 8. 最佳实践

### 8.1 数据选择

1. **云量控制**：选择云量<10%的影像
2. **季节选择**：根据研究目的选择季节
   - 植被监测：生长季
   - 水体监测：枯水期
3. **空间分辨率**：根据研究尺度选择

### 8.2 处理要点

1. **坐标系统一**：确保所有数据使用同一CRS
2. **大气校正**：定量研究必须进行
3. **质量控制**：检查云掩膜和异常值

### 8.3 精度验证

1. 混淆矩阵
2. Kappa系数
3. 总体精度
4. 用户精度/生产者精度
