"""
GIS/RS 知识库 RAG 检索管道
结构化、原子化、指令化的知识库管理

知识库目录结构:
- 01_Environment.md    : 底层运行环境与依赖隔离规范
- 02_GIS_Core.md       : 矢量与栅格数据的标准处理范式
- 03_GeoAI_Compute.md  : 空间张量计算与深度学习集成
- 04_Agent_Protocols.md: LangChain 系统提示词与工具约束
- 08_SelfCorrecting_REPL.md: 自修正 Python 代码执行系统（run_python_code 完整指南）
"""

import os
import json
from pathlib import Path
from typing import List, Optional, Dict, Any

# 可选依赖检查
try:
    from langchain.text_splitter import MarkdownHeaderTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("Warning: langchain 未安装，将使用简单关键词检索")


# =============================================================================
# 知识库管理器
# =============================================================================

class GISKnowledgeBase:
    """
    GIS/RS 知识库管理器
    
    提供结构化、原子化的知识检索能力，支持：
    - MarkdownHeaderTextSplitter 语义切分
    - FAISS 向量检索
    - 关键词备用检索
    """
    
    # 知识库文件映射
    KB_FILES = {
        # 基础规范
        "environment": "01_Environment.md",
        "gis_core": "02_GIS_Core.md",
        "geoai_compute": "03_GeoAI_Compute.md",
        "agent_protocols": "04_Agent_Protocols.md",
        # 新增理论 + Python 生态
        "gis_theory": "05_GIS_Theory.md",          # GIS/RS 理论（空间模型、CRS、遥感物理）
        "python_ecosystem": "06_Python_Ecosystem.md", # Python 生态与云原生遥感
        "advanced_qa": "07_Advanced_QA.md",          # 进阶专业知识问答 + STAC
        "self_repl": "08_SelfCorrecting_REPL.md",  # 自修正 Python 代码执行系统
        "comprehensive": "08_GIS_RS_Comprehensive.md",  # 综合技术手册（覆盖 A~L 全章节，2万+字）
    }
    
    def __init__(
        self,
        kb_dir: Optional[str] = None,
        embeddings_model: str = "text-embedding-3-small",
        vectorstore_path: Optional[str] = None
    ):
        """
        初始化知识库
        
        Args:
            kb_dir: 知识库目录路径，默认使用包内知识库
            embeddings_model: Embedding 模型名称
            vectorstore_path: FAISS 索引缓存路径
        """
        if kb_dir is None:
            kb_dir = Path(__file__).parent
        
        self.kb_dir = Path(kb_dir)
        self.embeddings_model = embeddings_model
        self.vectorstore_path = vectorstore_path
        
        self._vectorstore = None
        self._documents = []
        self._load_knowledge_base()
    
    def _load_knowledge_base(self):
        """加载知识库文档"""
        all_docs = []
        
        for kb_name, filename in self.KB_FILES.items():
            filepath = self.kb_dir / filename
            if filepath.exists():
                content = filepath.read_text(encoding='utf-8')
                self._documents.append({
                    "name": kb_name,
                    "filename": filename,
                    "content": content,
                    "filepath": str(filepath)
                })
                print(f"Loaded: {filename}")
        
        if LANGCHAIN_AVAILABLE and self._documents:
            self._build_vectorstore()
    
    def _build_vectorstore(self):
        """构建向量索引"""
        try:
            # 使用 MarkdownHeaderTextSplitter 保持语义块完整
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            
            # 切分所有文档
            langchain_docs = []
            for doc in self._documents:
                splits = splitter.split_text(doc["content"])
                for split in splits:
                    # 添加元数据
                    split.metadata = {
                        "source": doc["filename"],
                        "category": doc["name"]
                    }
                    langchain_docs.append(split)
            
            # 构建向量索引
            embeddings = OpenAIEmbeddings(model=self.embeddings_model)
            self._vectorstore = FAISS.from_documents(
                langchain_docs, 
                embeddings
            )
            
            # 保存索引
            if self.vectorstore_path:
                self._vectorstore.save_local(self.vectorstore_path)
            
            print(f"Vector store built with {len(langchain_docs)} chunks")
            
        except Exception as e:
            print(f"Vector store build failed: {e}")
            self._vectorstore = None
    
    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """关键词备用检索"""
        results = []
        query_lower = query.lower()

        # 增强关键词映射（按知识域分类）
        keyword_map = {
            # 基础规范
            "crs": ["crs", "坐标系", "投影", "坐标系转换", "gcs", "pcs", "epsg", "utm"],
            "raster": ["raster", "栅格", "tif", "tiff", "ndvi", "ndwi", "卫星影像", "landsat", "sentinel"],
            "vector": ["vector", "矢量", "shp", "geopandas", "geojson", "shapefile", "要素类"],
            "memory": ["memory", "oom", "内存", "大文件", "out of memory", "分块读取"],
            "folium": ["folium", "地图", "热力图", "html", "交互式地图", "leaflet"],
            "matplotlib": ["matplotlib", "绑图", "savefig", "静态图"],
            "torchgeo": ["torchgeo", "深度学习", "pytorch", "cuda", "语义分割", "unet"],
            "sjoin": ["sjoin", "空间连接", "within", "intersects", "spatial join"],
            "buffer": ["buffer", "缓冲区", "buffer(500)"],
            "mask": ["mask", "掩膜", "裁剪", "clip", "masking"],
            # 新增理论层
            "gis_theory": [
                "什么是gis", "矢量模型", "栅格模型", "空间数据模型", "拓扑",
                "电磁波谱", "光谱特征", "大气窗口", "主动传感器", "被动传感器",
                "雷达", "sar", "lidar", "高光谱", "多光谱", "空间分辨率",
                "光谱分辨率", "时间分辨率", "辐射分辨率", "重访周期",
                "邻接性", "连通性", "邻近性", "重叠", "包含",
                "基准面", "椭球体", "投影变形", "utm", "wgs84", "mercator",
                "数字高程模型", "dem", "坡度", "坡向",
            ],
            "python_ecosystem": [
                "shapely", "fiona", "xarray", "rioxarray", "earthpy", "pystac",
                "cog", "云原生", "stac", "dask", "pyrosar", "laspy", "spectral python",
                "pysal", "esda", "莫兰指数", "lisa", "osmnx", "最短路径",
                "kepler", "geemap", "leafmap", "矢量瓦片", "数字孪生",
            ],
            "advanced_qa": [
                "为什么ndvi", "ndvi公式", "geojson和shapefile区别", "矢量瓦片",
                "深度学习遥感", "时序遥感", "landtrendr", "变化检测",
                "google earth engine", "gee", "迁移学习遥感",
            ],
            "self_repl": [
                "python代码执行", "run_python_code", "self_correct", "self-correction",
                "自修正", "代码循环", "迭代修复", "syntaxerror", "importerror",
                "nameerror", "typeerror", "attributeerror", "keyerror", "indexerror",
                "filenotfound", "memoryerror", "oom", "除零错误", "会话管理",
                "session_id", "reset_session", "沙盒执行", "sandbox",
                "收敛检测", "死循环", "converged", "debug python",
                "python调试", "代码自检", "auto-correct", "自动修正",
            ],
            # 综合技术手册关键词（新覆盖的全面知识域）
            "comprehensive": [
                # A: 几何底层
                "de-9im", "拓扑不变量", "shapely有效", "make_valid", "linestring",
                "polygon有效性", "strtree", "geos", "jts", "测量维度",
                "buffer操作", "单边缓冲", "simplify简化",
                # B: CRS
                "cgcs2000", "gcj-02", "bd-09", "高斯克里格", "utm分带",
                "椭球体", "大地水准面", "七参数", "三参数", "投影变形",
                "墨卡托变形", "极地投影", "坐标系转换精度",
                # C: 遥感物理
                "辐射定标", "大气校正", "dos", "lasrc", "表观反射率",
                "landsat8", "landsat9", "sentinel-2", "sentinel-1",
                "斑点滤波", "lee滤波器", "gamma-map", "SAR后向散射",
                "主动遥感", "被动遥感", "大气窗口表",
                # D: 空间分析
                "地统计学", "变异函数", "球状模型", "高斯模型", "克里金插值",
                "块金效应", "变程", "基台值", "探索性空间分析",
                "热点分析", "geary_c", "join_count", "空间权重矩阵",
                "行标准化", "k近邻权重",
                # E: 遥感指数
                "evi增强植被", "savi土壤调节", "gndvi", "ndre红边",
                "mndwi改进水体", "ndbi建筑", "ndmi湿度", "nbr燃烧比",
                "水体指数表", "植被指数阈值", "ndvi分类",
                # F: 点云
                "laslaz", "laspy", "点云分类", "CSF地面滤波",
                "LiDAR", "数字高程模型", "点云插值",
                # G: 网络分析
                "等时圈", "可达范围", "重力模型", "可达性指数",
                "p-median", "设施选址", "isochrone", "KNN路径",
                "dijkstra", "网络连通性", "可达性分析",
                # H: 数据格式
                "flatgeobuf", "云优化", "geoparquet", "COG转换",
                "投影下推", "列式存储", "Parquet",
                # I: 深度学习
                "语义分割", "目标检测", "变化检测", "时序遥感",
                "迁移学习", "pytorch", "图像分割", "unet",
                # J: 数据库
                "postgis", "空间查询", "ST_Intersects", "ST_Within",
                "ST_Buffer", "GiST索引", "BRIN索引", "空间索引",
                # K: STAC
                "pystac", "planetary computer", "云原生遥感",
                "COG读取", "stac搜索", "item_collection",
                # L: 工程实践
                "GDAL命令行", "gdalwarp", "gdal_translate", "重投影",
                "分块处理", "瓦片大小", "概览金字塔", "工具选型",
            ],
        }
        
        matched_categories = []
        for cat, keywords in keyword_map.items():
            if any(kw in query_lower for kw in keywords):
                matched_categories.append(cat)
        
        # 从匹配的分类中提取相关内容
        for doc in self._documents:
            content = doc["content"]
            
            # 计算相关度
            relevance = 0
            if doc["name"] in matched_categories:
                relevance += 10
            
            # 检查关键词出现次数
            for kw in query_lower.split():
                if len(kw) > 2:
                    relevance += content.lower().count(kw) * 0.5
            
            if relevance > 0:
                results.append({
                    "category": doc["name"],
                    "source": doc["filename"],
                    "content": content,
                    "relevance": relevance,
                    "type": "keyword_match"
                })
        
        # 按相关度排序
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索知识库
        
        Args:
            query: 检索查询
            top_k: 返回前 k 个结果
            
        Returns:
            检索结果列表，每项包含 category, source, content, relevance
        """
        results = []
        
        if LANGCHAIN_AVAILABLE and self._vectorstore:
            try:
                # 向量检索
                docs = self._vectorstore.similarity_search_with_score(query, k=top_k)
                
                for doc, score in docs:
                    results.append({
                        "category": doc.metadata.get("category", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content,
                        "relevance": 1 - score,  # 转换为相关度
                        "type": "vector_search"
                    })
            except Exception as e:
                print(f"Vector search failed: {e}")
        
        # 如果向量检索无结果或不足，使用关键词检索
        if len(results) < top_k:
            keyword_results = self._keyword_search(query, top_k)
            
            # 合并结果，避免重复
            existing_sources = {r["source"] + r["content"][:50] for r in results}
            for kr in keyword_results:
                key = kr["source"] + kr["content"][:50]
                if key not in existing_sources:
                    results.append(kr)
        
        # 排序
        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]
    
    def get_reference_code(self, task: str) -> Optional[str]:
        """
        获取任务对应的参考代码
        
        Args:
            task: 任务描述 (如 "NDVI计算", "CRS转换")
            
        Returns:
            参考代码片段
        """
        results = self.search(task, top_k=1)
        
        if results:
            content = results[0]["content"]
            # 提取代码块
            if "```python" in content:
                start = content.find("```python") + 9
                end = content.find("```", start)
                if end > start:
                    return content[start:end].strip()
            
            # 如果没有代码块，返回文本描述
            return content[:500] + "..." if len(content) > 500 else content
        
        return None
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """
        格式化检索结果为可读文本
        """
        if not results:
            return "未找到相关知识库内容"

        output = []
        # 分类中文映射
        category_labels = {
            "environment": "环境与依赖",
            "gis_core": "GIS 核心处理",
            "geoai_compute": "GeoAI 计算",
            "agent_protocols": "Agent 协议",
            "gis_theory": "GIS/RS 理论",
            "python_ecosystem": "Python 生态",
            "advanced_qa": "进阶专业知识",
            "self_repl": "自修正代码执行",
            "comprehensive": "综合技术手册",  # 覆盖 A~L 全章节，2万+字知识库
            "unknown": "未知分类",
        }
        for i, r in enumerate(results, 1):
            cat_label = category_labels.get(r["category"], r["category"])
            output.append(f"\n{'=' * 60}")
            output.append(f"[{i}] {cat_label} | {r['source']}")
            output.append(f"{'=' * 60}")
            output.append(r["content"][:1500])  # 限制长度

            if len(r["content"]) > 1500:
                output.append("\n... (已截断)")

        return "\n".join(output)


# =============================================================================
# 检索工具工厂函数
# =============================================================================

def create_gis_retriever_tool(vectorstore=None, name: str = "search_gis_knowledge", description: str = None):
    """
    创建 GIS 知识库检索工具
    
    用法:
        from geoagent.knowledge import create_gis_retriever_tool
        
        tool = create_gis_retriever_tool()
        # 或
        tool = create_gis_retriever_tool(
            name="my_gis_search",
            description="当你不确定如何使用某个 GIS 库时使用此工具"
        )
    """
    if description is None:
        description = (
            "当且仅当你不知道如何使用 Geopandas, Rasterio, TorchGeo 等库编写空间处理代码，"
            "或者遇到 CRS、OOM 内存溢出、CUDA 加速问题时，使用此工具检索标准代码范例。"
            "输入应该是用中文描述你需要的 GIS 功能，如 'NDVI计算代码' 或 'CRS投影转换示例'。"
        )
    
    if LANGCHAIN_AVAILABLE:
        try:
            from langchain.tools.retriever import create_retriever_tool
            from langchain.vectorstores import FAISS
            
            # 如果没有提供 vectorstore，创建默认的
            if vectorstore is None:
                kb = GISKnowledgeBase()
                if kb._vectorstore:
                    vectorstore = kb._vectorstore
            
            if vectorstore:
                return create_retriever_tool(
                    vectorstore.as_retriever(search_kwargs={"k": 2}),
                    name,
                    description
                )
        except ImportError as e:
            print(f"Cannot create retriever tool: {e}")
    
    # 返回简单的函数工具作为备选
    def search_knowledge(query: str) -> str:
        """简单的知识检索函数"""
        kb = GISKnowledgeBase()
        results = kb.search(query, top_k=2)
        return kb.format_results(results)
    
    return search_knowledge


# =============================================================================
# 单例实例
# =============================================================================

# 全局知识库实例
_kb_instance: Optional[GISKnowledgeBase] = None


def get_knowledge_base() -> GISKnowledgeBase:
    """获取全局知识库实例"""
    global _kb_instance
    if _kb_instance is None:
        _kb_instance = GISKnowledgeBase()
    return _kb_instance


def search_gis_knowledge(query: str, top_k: int = 3) -> str:
    """
    便捷检索函数
    
    用法:
        from geoagent.knowledge import search_gis_knowledge
        
        result = search_gis_knowledge("如何计算NDVI")
        print(result)
    """
    kb = get_knowledge_base()
    results = kb.search(query, top_k=top_k)
    return kb.format_results(results)


# =============================================================================
# 示例用法
# =============================================================================

if __name__ == "__main__":
    # 示例检索
    print("=" * 60)
    print("GIS 知识库检索示例")
    print("=" * 60)
    
    kb = GISKnowledgeBase()
    
    # 测试查询
    queries = [
        "如何计算 NDVI",
        "CRS 坐标系不一致怎么办",
        "大文件 TIFF 内存溢出"
    ]
    
    for q in queries:
        print(f"\n查询: {q}")
        print("-" * 40)
        results = kb.search(q)
        print(kb.format_results(results))
        print()
