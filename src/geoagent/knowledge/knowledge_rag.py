"""
GIS/RS 知识库 RAG 检索管道
结构化、原子化、指令化的知识库管理

知识库目录结构:
- 01_Environment.md    : 底层运行环境与依赖隔离规范
- 02_GIS_Core.md       : 矢量与栅格数据的标准处理范式
- 03_GeoAI_Compute.md  : 空间张量计算与深度学习集成
- 04_Agent_Protocols.md: LangChain 系统提示词与工具约束
- 08_SelfCorrecting_REPL.md: 自修正 Python 代码执行系统
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


# =============================================================================
# 知识库管理器（增强版：静态记忆 + 动态 Workspace State）
# =============================================================================

class GISKnowledgeBase:
    """
    GIS/RS 知识库管理器

    提供结构化、原子化的知识检索能力，支持：
    - MarkdownHeaderTextSplitter 语义切分
    - FAISS 向量检索
    - 关键词备用检索
    - **Workspace State 动态上下文注入**（第三层架构核心）
    """

    # 知识库文件映射
    KB_FILES = {
        "environment": "01_Environment.md",
        "gis_core": "02_GIS_Core.md",
        "geoai_compute": "03_GeoAI_Compute.md",
        "agent_protocols": "04_Agent_Protocols.md",
        "gis_theory": "05_GIS_Theory.md",
        "python_ecosystem": "06_Python_Ecosystem.md",
        "advanced_qa": "07_Advanced_QA.md",
        "self_repl": "08_SelfCorrecting_REPL.md",
        "comprehensive": "09_GIS_RS_Comprehensive.md",
    }

    def __init__(
        self,
        kb_dir: Optional[str] = None,
        embeddings_model: str = "text-embedding-3-small",
        vectorstore_path: Optional[str] = None
    ):
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

        if LANGCHAIN_AVAILABLE and self._documents:
            self._build_vectorstore()

    def _build_vectorstore(self):
        """构建向量索引"""
        if not LANGCHAIN_AVAILABLE:
            return

        try:
            # 检查 API key 是否可用
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
            if not api_key:
                print("警告: 未设置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY，向量检索将使用关键词搜索降级")
                return

            from langchain.text_splitter import MarkdownHeaderTextSplitter
            from langchain_community.vectorstores import FAISS
            from langchain_openai import OpenAIEmbeddings

            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ]
            splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

            langchain_docs = []
            for doc in self._documents:
                splits = splitter.split_text(doc["content"])
                for split in splits:
                    split.metadata = {
                        "source": doc["filename"],
                        "category": doc["name"]
                    }
                    langchain_docs.append(split)

            embeddings = OpenAIEmbeddings(model=self.embeddings_model)
            self._vectorstore = FAISS.from_documents(
                langchain_docs,
                embeddings
            )

            if self.vectorstore_path:
                self._vectorstore.save_local(self.vectorstore_path)

        except Exception as e:
            print(f"向量索引构建失败: {e}")
            self._vectorstore = None

    def _keyword_search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """关键词备用检索"""
        results = []
        query_lower = query.lower()

        keyword_map = {
            "crs": ["crs", "坐标系", "投影", "坐标系转换", "gcs", "pcs", "epsg", "utm"],
            "raster": ["raster", "栅格", "tif", "tiff", "ndvi", "ndwi", "卫星影像", "landsat", "sentinel"],
            "vector": ["vector", "矢量", "shp", "geopandas", "geojson", "shapefile", "要素类"],
            "memory": ["memory", "oom", "内存", "大文件", "out of memory", "分块读取"],
            "folium": ["folium", "地图", "热力图", "html", "交互式地图"],
            "matplotlib": ["matplotlib", "绑图", "savefig", "静态图"],
            "torchgeo": ["torchgeo", "深度学习", "pytorch", "cuda", "语义分割", "unet"],
            "sjoin": ["sjoin", "空间连接", "within", "intersects", "spatial join"],
            "buffer": ["buffer", "缓冲区", "buffer(500)"],
            "mask": ["mask", "掩膜", "裁剪", "clip", "masking"],
            "gis_theory": [
                "什么是gis", "矢量模型", "栅格模型", "空间数据模型", "拓扑",
                "电磁波谱", "光谱特征", "大气窗口", "主动传感器", "被动传感器",
                "雷达", "sar", "lidar", "高光谱", "多光谱", "空间分辨率",
                "基准面", "椭球体", "投影变形", "utm", "wgs84", "mercator",
                "数字高程模型", "dem", "坡度", "坡向",
            ],
            "python_ecosystem": [
                "shapely", "fiona", "xarray", "rioxarray", "earthpy", "pystac",
                "cog", "云原生", "stac", "dask", "pysal", "esda", "莫兰指数", "lisa", "osmnx",
            ],
            "advanced_qa": [
                "为什么ndvi", "ndvi公式", "geojson和shapefile区别", "矢量瓦片",
                "深度学习遥感", "时序遥感", "变化检测",
            ],
            "self_repl": [
                "python代码执行", "run_python_code", "自修正", "代码循环",
                "迭代修复", "syntaxerror", "importerror", "nameerror",
                "converged", "debug python", "代码自检",
            ],
            "comprehensive": [
                "de-9im", "拓扑不变量", "shapely有效", "make_valid",
                "cgcs2000", "gcj-02", "bd-09", "高斯克里格",
                "辐射定标", "大气校正", "landsat8", "sentinel-2",
                "地统计学", "变异函数", "克里金插值",
                "evi", "savi", "gndvi", "mndwi", "ndbi",
                "laslaz", "点云分类", "CSF地面滤波",
                "等时圈", "可达范围", "重力模型", "isochrone",
                "flatgeobuf", "geoparquet", "COG转换",
                "postgis", "ST_Intersects", "ST_Buffer",
                "pystac", "planetary computer",
                "gdalwarp", "gdal_translate", "分块处理",
            ],
        }

        matched_categories = []
        for cat, keywords in keyword_map.items():
            if any(kw in query_lower for kw in keywords):
                matched_categories.append(cat)

        for doc in self._documents:
            content = doc["content"]

            relevance = 0
            if doc["name"] in matched_categories:
                relevance += 10

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

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        检索知识库（向量检索 + 关键词备用）
        """
        results = []

        if LANGCHAIN_AVAILABLE and self._vectorstore:
            try:
                docs = self._vectorstore.similarity_search_with_score(query, k=top_k)

                for doc, score in docs:
                    results.append({
                        "category": doc.metadata.get("category", "unknown"),
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content,
                        "relevance": 1 - score,
                        "type": "vector_search"
                    })
            except Exception as e:
                print(f"Vector search failed: {e}")

        if len(results) < top_k:
            keyword_results = self._keyword_search(query, top_k)
            existing_sources = {r["source"] + r["content"][:50] for r in results}
            for kr in keyword_results:
                key = kr["source"] + kr["content"][:50]
                if key not in existing_sources:
                    results.append(kr)

        results.sort(key=lambda x: x["relevance"], reverse=True)
        return results[:top_k]

    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """格式化检索结果为可读文本"""
        if not results:
            return "未找到相关知识库内容"

        output = []
        category_labels = {
            "environment": "环境与依赖",
            "gis_core": "GIS 核心处理",
            "geoai_compute": "GeoAI 计算",
            "agent_protocols": "Agent 协议",
            "gis_theory": "GIS/RS 理论",
            "python_ecosystem": "Python 生态",
            "advanced_qa": "进阶专业知识",
            "self_repl": "自修正代码执行",
            "comprehensive": "综合技术手册",
            "unknown": "未知分类",
        }

        for i, r in enumerate(results, 1):
            cat_label = category_labels.get(r["category"], r["category"])
            output.append(f"\n{'=' * 60}")
            output.append(f"[{i}] {cat_label} | {r['source']}")
            output.append(f"{'=' * 60}")
            output.append(r["content"][:1500])

            if len(r["content"]) > 1500:
                output.append("\n... (已截断)")

        return "\n".join(output)


# =============================================================================
# Workspace State 获取（第三层：动态记忆核心）
# =============================================================================

def get_workspace_state() -> str:
    """
    扫描 workspace/ 目录，生成文件状态摘要。

    这是"动态记忆"的核心 — 在 Agent 每次生成代码前调用，
    根治大模型的"文件幻觉"。

    示例输出：
    ```
    [当前工作区文件状态 — 已上传到 workspace/ 的 GIS 数据文件]：
      - `study_area.shp` [矢量(Shapefile)] (23.5 KB)
      - `ndvi.tif` [栅格(GeoTIFF)] (15.2 MB)
    ⚠️ 重要提醒：以上文件是当前工作区中真实存在的文件。
    ```
    """
    try:
        from geoagent.gis_tools.fixed_tools import list_workspace_files
    except Exception:
        # 备用实现
        try:
            ws = Path(__file__).parent.parent.parent / "workspace"
            if not ws.exists():
                return "[当前工作区文件：无]\n"
            all_files = []
            for ext in ["*.shp", "*.geojson", "*.json", "*.gpkg", "*.parquet",
                        "*.tif", "*.tiff", "*.cog", "*.las", "*.laz",
                        "*.html", "*.csv"]:
                all_files.extend(ws.glob(ext))
            files = [f.name for f in all_files]
        except Exception:
            return "[当前工作区文件：未知（读取失败）]\n"

    files = list_workspace_files()
    if not files:
        return "[当前工作区文件：无]\n"

    lines = ["[当前工作区文件状态 — 已上传到 workspace/ 的 GIS 数据文件]：\n"]
    for fname in files:
        ext = Path(fname).suffix.lower()
        type_label = {
            ".shp": "矢量(Shapefile)", ".geojson": "矢量(GeoJSON)",
            ".json": "矢量(GeoJSON)", ".gpkg": "矢量(GeoPackage)",
            ".parquet": "矢量(GeoParquet)", ".tif": "栅格(GeoTIFF)",
            ".tiff": "栅格(GeoTIFF)", ".cog": "栅格(COG)",
            ".las": "点云(LAS)", ".laz": "点云(LAZ)",
        }.get(ext, f"其他({ext})")

        size = ""
        ws = Path(__file__).parent.parent.parent / "workspace"
        fpath = ws / fname
        if fpath.exists():
            kb = fpath.stat().st_size / 1024
            size = f" ({kb:.1f} KB)" if kb < 1024 else f" ({kb/1024:.1f} MB)"

        lines.append(f"  - `{fname}` [{type_label}]{size}")

    lines.append(
        "\n⚠️ **重要提醒**：以上文件是当前工作区中**真实存在**的文件。"
        "在生成代码时，你**必须**使用这些文件名，"
        "**禁止**凭空臆造不存在的文件名！"
    )
    return "\n".join(lines)


# =============================================================================
# 检索工具工厂函数
# =============================================================================

def create_gis_retriever_tool(vectorstore=None, name: str = "search_gis_knowledge", description: str = None):
    """
    创建 GIS 知识库检索工具
    
    注意：当 LangChain 可用时返回 Tool 对象，否则返回函数（兼容旧代码）
    """
    if description is None:
        description = (
            "当且仅当你不知道如何使用 Geopandas, Rasterio, TorchGeo 等库编写空间处理代码，"
            "或者遇到 CRS、OOM 内存溢出、CUDA 加速问题时，使用此工具检索标准代码范例。"
            "输入应该是用中文描述你需要的 GIS 功能。"
        )

    if LANGCHAIN_AVAILABLE:
        try:
            from langchain.tools.retriever import create_retriever_tool
            from langchain.vectorstores import FAISS

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
        except Exception as e:
            print(f"Failed to create retriever tool: {e}")

    # 返回一个简单的结构化工具（用于非 LangChain 环境）
    # 创建一个简单的 callable 类来替代 LangChain Tool
    class SimpleSearchTool:
        """简单的知识库检索工具（兼容非 LangChain 环境）"""
        name = name
        description = description
        
        def invoke(self, query: str) -> str:
            kb = GISKnowledgeBase()
            results = kb.search(query, top_k=2)
            return kb.format_results(results)
        
        def __call__(self, query: str) -> str:
            return self.invoke(query)
    
    return SimpleSearchTool()


# =============================================================================
# 单例实例
# =============================================================================

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


if __name__ == "__main__":
    print("=" * 60)
    print("GIS 知识库检索示例")
    print("=" * 60)

    kb = GISKnowledgeBase()

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

    print("\n" + "=" * 60)
    print("Workspace State 示例：")
    print("=" * 60)
    print(get_workspace_state())
