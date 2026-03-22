"""
测试数据探针（Data Profiler）功能
验证工作区空间文件 Schema 元数据提取和 Prompt 注入的准确性
"""

import pytest
from pathlib import Path


class TestSniffSpatialFile:
    """测试单个空间文件的探针功能"""

    def test_sniff_with_nonexistent_file(self):
        """测试不存在的文件返回失败结构"""
        from geoagent.gis_tools.data_profiler import sniff_spatial_file

        result = sniff_spatial_file(Path("/nonexistent/file.shp"))
        assert result["success"] is False
        assert "file_name" in result
        assert "error" in result

    def test_sniff_with_unsupported_format(self):
        """测试不支持的文件格式"""
        from geoagent.gis_tools.data_profiler import sniff_spatial_file

        # 创建一个临时文件，扩展名不支持
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"dummy data")
            temp_path = Path(f.name)

        try:
            result = sniff_spatial_file(temp_path)
            assert result["success"] is False
            assert "不支持的文件格式" in result["error"]
        finally:
            os.unlink(temp_path)


class TestSniffWorkspaceDir:
    """测试工作区目录批量探针"""

    def test_sniff_empty_workspace(self, tmp_path):
        """测试空目录返回空列表"""
        from geoagent.gis_tools.data_profiler import sniff_workspace_dir

        result = sniff_workspace_dir(tmp_path)
        assert result == []

    def test_sniff_nonexistent_workspace(self):
        """测试不存在的目录返回空列表"""
        from geoagent.gis_tools.data_profiler import sniff_workspace_dir

        result = sniff_workspace_dir(Path("/nonexistent/workspace"))
        assert result == []


class TestBuildWorkspaceProfileBlock:
    """测试 Profile Block 构建"""

    def test_build_with_empty_profiles(self):
        """测试空 profile 返回空字符串"""
        from geoagent.gis_tools.data_profiler import build_workspace_profile_block

        result = build_workspace_profile_block([])
        assert result == ""

    def test_build_with_profiles(self):
        """测试正常 profile 块构建"""
        from geoagent.gis_tools.data_profiler import build_workspace_profile_block

        profiles = [
            {
                "success": True,
                "file_name": "test.shp",
                "file_type": "vector",
                "geometry_type": "Polygon",
                "crs": "EPSG:4326",
                "columns": ["id", "name"],
                "column_types": {"id": "整数", "name": "文本"},
                "sample_data": [{"id": 1, "name": "test"}],
            }
        ]

        result = build_workspace_profile_block(profiles)

        # 验证包含关键信息
        assert "【工作区文件详细情报】" in result
        assert "test.shp" in result
        assert "Polygon" in result
        assert "EPSG:4326" in result
        assert "id" in result
        assert "name" in result
        assert "整数" in result
        assert "文本" in result
        assert "数据样例" in result
        assert "【铁律】" in result
        # 铁律内容
        assert "属性筛选" in result
        assert "字段名" in result

    def test_build_with_raster_profile(self):
        """测试栅格文件 profile"""
        from geoagent.gis_tools.data_profiler import build_workspace_profile_block

        profiles = [
            {
                "success": True,
                "file_name": "dem.tif",
                "file_type": "raster",
                "geometry_type": "Raster",
                "crs": "EPSG:32650",
                "columns": [],
                "column_types": {},
                "sample_data": [],
                "raster_info": {
                    "band_count": 1,
                    "resolution": "30.00 x 30.00",
                    "dtypes": ["float32"],
                },
            }
        ]

        result = build_workspace_profile_block(profiles)

        assert "dem.tif" in result
        assert "Raster" in result
        assert "波段数: 1" in result
        assert "分辨率: 30.00 x 30.00" in result


class TestQuickProfile:
    """测试快速探针入口"""

    def test_quick_profile_empty_workspace(self, tmp_path):
        """测试空工作区返回空字符串"""
        from geoagent.gis_tools.data_profiler import quick_profile

        result = quick_profile(tmp_path)
        assert result == ""


class TestCacheMechanism:
    """测试缓存机制"""

    def test_cached_result_is_list(self):
        """测试带缓存的结果是列表"""
        from geoagent.gis_tools.data_profiler import sniff_workspace_dir_cached

        # 两次调用应该返回相同的结果
        result1 = sniff_workspace_dir_cached()
        result2 = sniff_workspace_dir_cached()

        assert isinstance(result1, list)
        assert result1 == result2

    def test_clear_cache(self):
        """测试清除缓存"""
        from geoagent.gis_tools.data_profiler import (
            sniff_workspace_dir_cached,
            clear_profiler_cache,
        )

        # 先获取一次
        result1 = sniff_workspace_dir_cached()

        # 清除缓存
        clear_profiler_cache()

        # 清除后再次调用应正常返回（内部会重新扫描）
        result2 = sniff_workspace_dir_cached()
        assert isinstance(result2, list)


class TestLayer3Integration:
    """测试 layer3_orchestrate 集成"""

    def test_build_profile_block_function_exists(self):
        """测试 _build_workspace_profile_block 函数存在"""
        from geoagent.layers.layer3_orchestrate import _build_workspace_profile_block

        result = _build_workspace_profile_block()
        assert isinstance(result, str)

    def test_scan_workspace_files_extended(self):
        """测试扩展后的 _scan_workspace_files 返回额外字段"""
        from geoagent.layers.layer3_orchestrate import _scan_workspace_files

        results = _scan_workspace_files()

        # 如果有结果，检查字段
        if results:
            for r in results:
                # 原有字段
                assert "file_name" in r
                assert "file_type" in r
                assert "columns" in r
                # 新增字段
                assert "dtypes" in r
                assert "column_types" in r
                assert "text_columns" in r
                assert "sample_data" in r
                assert "crs" in r


class TestFixedToolsIntegration:
    """测试 fixed_tools 集成"""

    def test_sniff_workspace_profiler_exists(self):
        """测试 sniff_workspace_profiler 工具函数存在"""
        from geoagent.gis_tools.fixed_tools import sniff_workspace_profiler

        result = sniff_workspace_profiler()
        assert isinstance(result, str)

    def test_sniff_workspace_profiler_raw_exists(self):
        """测试 sniff_workspace_profiler_raw 工具函数存在"""
        from geoagent.gis_tools.fixed_tools import sniff_workspace_profiler_raw

        result = sniff_workspace_profiler_raw()
        assert isinstance(result, str)
        # 应该是 JSON 格式
        import json
        try:
            parsed = json.loads(result)
            assert isinstance(parsed, list)
        except json.JSONDecodeError:
            pytest.fail("sniff_workspace_profiler_raw 应返回有效的 JSON")

    def test_clear_workspace_cache_exists(self):
        """测试 clear_workspace_cache 工具函数存在"""
        from geoagent.gis_tools.fixed_tools import clear_workspace_cache

        result = clear_workspace_cache()
        assert isinstance(result, str)


class TestDtypeMapping:
    """测试 dtype 映射"""

    def test_dtype_to_label_function(self):
        """测试 _dtype_to_label 函数"""
        from geoagent.layers.layer3_orchestrate import _dtype_to_label

        assert _dtype_to_label("int64") == "整数"
        assert _dtype_to_label("float64") == "浮点数"
        assert _dtype_to_label("object") == "文本"
        assert _dtype_to_label("bool") == "布尔"
        assert _dtype_to_label("datetime64[ns]") == "日期时间"
        assert _dtype_to_label("unknown_type") == "未知"
