# 空间计算与深度学习

## 空间张量计算

### 核心逻辑

将地理空间数据转换为可用于 GPU 加速的张量（Tensors）。

### PyTorch 与 CUDA 整合

**强制规范**：检查硬件加速并在数据流转中保持设备一致性。

```python
import torch
import numpy as np
import rasterio

# 设备检测
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 从栅格读取张量
def raster_to_tensor(raster_path, band_index=1):
    """将栅格波段转换为 PyTorch 张量"""
    with rasterio.open(raster_path) as src:
        data = src.read(band_index).astype(np.float32)
    
    tensor = torch.from_numpy(data).unsqueeze(0)  # 添加通道维度
    return tensor.to(device)

# 从数组创建张量
def array_to_tensor(arr):
    """将 NumPy 数组转换为 PyTorch 张量并移到 GPU"""
    tensor = torch.from_numpy(arr.astype(np.float32))
    if len(tensor.shape) == 2:
        tensor = tensor.unsqueeze(0)  # [H, W] -> [1, H, W]
    return tensor.to(device)

# 计算后移回 CPU
def tensor_to_array(tensor):
    """将张量转回 NumPy 数组"""
    return tensor.cpu().detach().numpy()
```

---

## TorchGeo 集成

### Sentinel-2 影像加载

```python
import torch
from torchgeo.datasets import Sentinel2
from torchgeo.samplers import RandomGeoSampler
from pathlib import Path

# 数据集路径
data_dir = Path('workspace/sentinel_data')

# 加载 Sentinel-2 数据
dataset = Sentinel2(root=data_dir)

# 随机采样
sampler = RandomGeoSampler(dataset.size, size=256, length=100)
dataloader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=4)

# 训练循环
for batch in dataloader:
    # batch 包含图像和标签
    images = batch['image']  # Shape: [B, C, H, W]
    labels = batch['mask'] if 'mask' in batch else None
    
    # 确保在正确设备上
    images = images.to(device)
    
    # 前向传播
    # ... 你的模型代码 ...
    
    print(f"Batch shape: {images.shape}, Device: {images.device}")
```

### 自定义遥感数据集

```python
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np
from pathlib import Path

class RemoteSensingDataset(Dataset):
    """自定义遥感数据集"""
    
    def __init__(self, raster_paths, patch_size=256, transform=None):
        self.raster_paths = raster_paths
        self.patch_size = patch_size
        self.transform = transform
    
    def __len__(self):
        return len(self.raster_paths)
    
    def __getitem__(self, idx):
        with rasterio.open(self.raster_paths[idx]) as src:
            image = src.read().astype(np.float32)
        
        # 归一化
        image = self._normalize(image)
        
        # 随机裁剪
        h, w = image.shape[1:]
        top = np.random.randint(0, h - self.patch_size)
        left = np.random.randint(0, w - self.patch_size)
        
        image = image[:, top:top+self.patch_size, left:left+self.patch_size]
        
        # 转为张量
        image = torch.from_numpy(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image
    
    def _normalize(self, image):
        """逐波段归一化"""
        for i in range(image.shape[0]):
            band = image[i]
            if band.max() > band.min():
                image[i] = (band - band.min()) / (band.max() - band.min())
        return image
```

---

## 图像分割模型

### 使用 segmentation_models_pytorch

```python
import torch
import segmentation_models_pytorch as smp

# 创建模型
model = smp.Unet(
    encoder_name='resnet34',
    encoder_weights='imagenet',
    in_channels=4,  # NIR + RGB
    classes=2  # 二分类
)

# 移到 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# 推理
def predict(model, image_tensor):
    """推理函数"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        preds = torch.argmax(probs, dim=1)
    return preds.cpu(), probs.cpu()

# 使用示例
# 假设 image_tensor shape: [1, 4, 256, 256]
preds, probs = predict(model, image_tensor)
print(f"预测结果形状: {preds.shape}")
```

---

## 空间统计与机器学习

### Libpysal 集成

```python
import geopandas as gpd
from libpysal.weights import Queen, KNN
from esda.moran import Moran, Moran_Local
import numpy as np

# 读取数据
gdf = gpd.read_file('workspace/data.shp')

# 统一 CRS
gdf = gdf.to_crs('EPSG:3857')

# 构建空间权重矩阵
w_queen = Queen.from_dataframe(gdf)
w_queen.transform = 'r'  # 行标准化

# 全局空间自相关 (Moran's I)
y = gdf['target_column'].values
moran = Moran(y, w_queen)
print(f"Moran's I: {moran.I:.4f}")
print(f"P-value: {moran.p_sim:.4f}")

# 局部空间自相关 (LISA)
lisa = Moran_Local(y, w_queen)
gdf['lisa_q'] = lisa.q  # 象限 (HH, HL, LH, LL)
gdf['lisa_p'] = lisa.p_sim  # P 值
gdf['lisa_sig'] = lisa.Is  # 局部 Moran's I

# 保存结果
gdf.to_file('workspace/lisa_results.shp')
print("LISA 分析结果已保存")
```

### 空间聚类

```python
import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np

# 读取点数据
gdf = gpd.read_file('workspace/pois.shp')

# 提取坐标
coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])

# DBSCAN 聚类
db = DBSCAN(eps=1000, min_samples=5).fit(coords)  # 1000米范围内至少5个点

# 添加聚类标签
gdf['cluster'] = db.labels_

# 统计每个聚类的数量
cluster_counts = gdf.groupby('cluster').size()
print(cluster_counts)

# 保存
gdf.to_file('workspace/clustered_pois.shp')
```

---

## 大规模栅格处理

### Dask 集成

```python
import rioxarray
import xarray as xr
import dask.array as da

# 使用 rioxarray 懒加载
rds = rioxarray.open_rasterio('workspace/large_dem.tif', chunks={'x': 1000, 'y': 1000})

print(f"数据形状: {rds.shape}")
print(f"分块大小: {rds.chunks}")

# 计算坡度（分块处理，不会内存溢出）
slope = rds.differentiate('x')**2 + rds.differentiate('y')**2
slope = (slope ** 0.5 * 111320 / 110540)  # 转换为度

# 保存结果
slope.rio.to_raster('outputs/slope.tif', chunksize=(1000, 1000))
print("坡度图已保存")
```

### 多时相分析

```python
import rioxarray
import numpy as np
import xarray as xr
from pathlib import Path

# 读取多时相影像
dates = ['2020_01.tif', '2020_06.tif', '2020_12.tif']
data_arrays = []

for date_file in dates:
    ds = rioxarray.open_rasterio(f'workspace/{date_file}', chunks={'x': 1000, 'y': 1000})
    data_arrays.append(ds.squeeze())

# 合并为 xarray DataArray
combined = xr.concat(data_arrays, dim='time')
combined = combined.assign_coords(time=np.arange(len(dates)))

# 计算时间序列统计
mean_val = combined.mean(dim='time')
std_val = combined.std(dim='time')

# NDVI 变化检测
ndvi_change = data_arrays[1] - data_arrays[0]
ndvi_change.rio.to_raster('outputs/ndvi_change.tif')
print("NDVI 变化检测完成")
```
