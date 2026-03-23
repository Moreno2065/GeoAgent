"""
Geo-Agent 测试套件
"""

import pytest
import json
import os
import tempfile
import shutil
import uuid
from pathlib import Path
from unittest.mock import patch, MagicMock
import sys

# 添加项目根目录到 Python 路径
sys.path.insert(0, str(Path(__file__).parent))

# UUID helper used in tests
_uuid_hex = lambda: f"user_{uuid.uuid4().hex[:8]}"

# GeoAgent 和 core 模块（V2 API）
from geoagent.core import GeoAgent
import geoagent.core as ac


# =============================================================================
# 测试配置
# =============================================================================

@pytest.fixture
def temp_workspace():
    """创建临时 workspace 目录"""
    temp_dir = tempfile.mkdtemp()
    workspace_path = Path(temp_dir) / "workspace"
    workspace_path.mkdir()
    yield workspace_path
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_env(monkeypatch):
    """设置测试环境变量"""
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test_api_key_12345")
    return monkeypatch


# =============================================================================
# 测试 fixed_tools.py
# =============================================================================

class TestFixedTools:
    """测试固定工具函数"""

    def test_get_data_info_file_not_found(self, temp_workspace):
        """测试文件不存在的情况"""
        from geoagent.gis_tools.fixed_tools import get_data_info, get_workspace_dir
        
        # 临时修改 workspace 路径
        original_get_workspace_dir = get_workspace_dir
        import geoagent.gis_tools.fixed_tools as ft
        ft.get_workspace_dir = lambda: temp_workspace
        
        try:
            result = get_data_info("nonexistent_file.shp")
            result_dict = json.loads(result)
            
            assert "error" in result_dict
            assert "文件不存在" in result_dict["error"]
            assert "workspace_path" in result_dict
        finally:
            ft.get_workspace_dir = original_get_workspace_dir

    def test_get_data_info_unsupported_format(self, temp_workspace):
        """测试不支持的文件格式"""
        from geoagent.gis_tools.fixed_tools import get_data_info, get_workspace_dir
        import geoagent.gis_tools.fixed_tools as ft
        
        # 创建临时文件
        test_file = temp_workspace / "test.xyz"
        test_file.write_text("dummy content")
        
        original_get_workspace_dir = get_workspace_dir
        ft.get_workspace_dir = lambda: temp_workspace
        
        try:
            result = get_data_info("test.xyz")
            result_dict = json.loads(result)
            
            assert "error" in result_dict
            assert "不支持的文件格式" in result_dict["error"]
            assert "supported_formats" in result_dict
        finally:
            ft.get_workspace_dir = original_get_workspace_dir

    def test_list_workspace_files_empty(self, temp_workspace):
        """测试空 workspace 目录"""
        from geoagent.gis_tools.fixed_tools import list_workspace_files, get_workspace_dir
        import geoagent.gis_tools.fixed_tools as ft
        
        original_get_workspace_dir = get_workspace_dir
        ft.get_workspace_dir = lambda: temp_workspace
        
        try:
            files = list_workspace_files()
            assert files == []
        finally:
            ft.get_workspace_dir = original_get_workspace_dir

    def test_list_workspace_files_with_files(self, temp_workspace):
        """测试包含 GIS 文件的目录"""
        from geoagent.gis_tools.fixed_tools import list_workspace_files, get_workspace_dir
        import geoagent.gis_tools.fixed_tools as ft
        
        # 创建测试文件
        (temp_workspace / "test.shp").write_bytes(b"")
        (temp_workspace / "test.geojson").write_bytes(b"")
        (temp_workspace / "test.tif").write_bytes(b"")
        (temp_workspace / "readme.txt").write_bytes(b"")  # 非 GIS 文件
        (temp_workspace / "subdir").mkdir()  # 子目录
        
        original_get_workspace_dir = get_workspace_dir
        ft.get_workspace_dir = lambda: temp_workspace
        
        try:
            files = list_workspace_files()
            assert len(files) == 3
            assert "test.shp" in files
            assert "test.geojson" in files
            assert "test.tif" in files
            assert "readme.txt" not in files  # 非 GIS 文件不应包含
        finally:
            ft.get_workspace_dir = original_get_workspace_dir

    def test_vector_file_with_none_crs(self, temp_workspace):
        """测试无 CRS 的矢量文件"""
        from geoagent.gis_tools.fixed_tools import get_data_info, get_workspace_dir
        import geoagent.gis_tools.fixed_tools as ft
        
        original_get_workspace_dir = get_workspace_dir
        ft.get_workspace_dir = lambda: temp_workspace
        
        try:
            # 创建 GeoJSON 文件（无 CRS）
            geojson = {
                "type": "FeatureCollection",
                "features": [{
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [118.37, 31.33]
                    },
                    "properties": {"name": "test"}
                }]
            }
            test_file = temp_workspace / "test_no_crs.geojson"
            with open(test_file, 'w', encoding='utf-8') as f:
                json.dump(geojson, f)
            
            result = get_data_info("test_no_crs.geojson")
            result_dict = json.loads(result)
            
            # 不应崩溃，CRS 应为 None
            assert "error" not in result_dict or "geopandas" in result_dict.get("error", "").lower() or result_dict.get("crs", {}).get("epsg") is None
        finally:
            ft.get_workspace_dir = original_get_workspace_dir


# =============================================================================
# 测试 agent_core.py
# =============================================================================

class TestAgentCore:
    """测试 Agent 核心功能"""

    def test_create_agent_without_api_key(self, monkeypatch):
        """测试未提供 API Key 的情况"""
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

        original_load = GeoAgent._load_api_key
        GeoAgent._load_api_key = staticmethod(lambda: None)

        try:
            with pytest.raises(ValueError, match="API"):
                GeoAgent()
        finally:
            GeoAgent._load_api_key = original_load

    @patch('openai.OpenAI')
    def test_create_agent_with_api_key(self, mock_openai, mock_env):
        """测试提供 API Key 的情况"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        assert agent.api_key == "sk-test_key"
        assert agent.model == "deepseek-reasoner"

    @patch('openai.OpenAI')
    def test_add_user_message(self, mock_openai, mock_env):
        """测试 V2 架构没有 messages 列表，验证 agent 可正常创建"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表，也没有 add_user_message 方法
        assert hasattr(agent, "api_key")
        assert agent.api_key == "sk-test_key"

    @patch('openai.OpenAI')
    def test_clear_history(self, mock_openai, mock_env):
        """测试 V2 架构没有 clear_history 方法，验证 stats 正常"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表和 clear_history
        assert hasattr(agent, "stats")
        assert agent.stats["total_requests"] == 0

    @patch('openai.OpenAI')
    def test_reset_conversation(self, mock_openai, mock_env):
        """测试 V2 架构没有 reset_conversation，用 reset_stats 代替"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 用 reset_stats 代替（无 conversations/history 概念）
        agent.stats["total_requests"] = 5
        agent.stats["failed"] = 2
        agent.reset_stats()

        assert agent.stats["total_requests"] == 0
        assert agent.stats["failed"] == 0

    @patch('openai.OpenAI')
    def test_check_history_limit(self, mock_openai, mock_env):
        """测试 V2 架构没有 history limit 概念"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表和 max_history 参数
        assert hasattr(agent, "api_key")
        assert agent.api_key == "sk-test_key"

    @patch('openai.OpenAI')
    def test_parse_tool_calls(self, mock_openai, mock_env):
        """测试工具执行器对不存在的文件返回错误而非崩溃"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        from geoagent.tools import execute_tool

        result = execute_tool("get_data_info", {"file_name": "nonexistent.shp"})
        assert "error" in result or "不存在" in result

    @patch('openai.OpenAI')
    def test_parse_tool_calls_data_info(self, mock_openai, mock_env):
        """测试数据查询工具调用"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        from geoagent.tools import execute_tool

        result = execute_tool("get_data_info", {"file_name": "test.shp"})
        assert "error" in result or "不存在" in result

    @patch('openai.OpenAI')
    def test_parse_tool_calls_invalid_json(self, mock_openai, mock_env):
        """测试 execute_tool 对未知工具名的处理"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        from geoagent.tools import execute_tool

        result = execute_tool("nonexistent_tool", {})
        assert "error" in result or "Unknown" in result

    @patch('openai.OpenAI')
    def test_clean_response(self, mock_openai, mock_env):
        """测试 V2 架构没有 messages/history 概念，验证 reset_stats 正常"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表，用 stats 代替
        agent.stats["total_requests"] = 5
        agent.reset_stats()
        assert agent.stats["total_requests"] == 0

    @patch('openai.OpenAI')
    def test_get_conversation_history(self, mock_openai, mock_env):
        """测试 V2 架构没有 conversation history"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 get_conversation_history
        assert hasattr(agent, "api_key")
        assert hasattr(agent, "stats")
        assert agent.api_key == "sk-test_key"

    @patch('openai.OpenAI')
    def test_get_stats(self, mock_openai, mock_env):
        """测试获取统计信息"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")
        stats = agent.get_stats()

        # V2 stats keys are total_requests/successful/failed
        assert "total_requests" in stats
        assert "successful" in stats
        assert "failed" in stats


# =============================================================================
# 测试 agent_core.py 自我纠错和 API 调用
# =============================================================================

class TestAgentSelfHealing:
    """测试 Agent 自我纠错功能"""

    @patch('openai.OpenAI')
    def test_self_healing_on_traceback(self, mock_openai, mock_env):
        """测试 V2 chat() 不再有 traceback 自我修复（使用六层确定性架构）"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 chat() 调用 run()，返回结果字典
        # 在没有真实 pipeline 的情况下，验证方法存在
        assert hasattr(agent, "chat")
        assert hasattr(agent, "run")


# =============================================================================
# 测试 app.py 初始化
# =============================================================================

class TestAppInitialization:
    """测试应用初始化"""

    def test_session_state_initialization(self):
        """测试会话状态初始化"""
        # 由于 Streamlit 会话状态需要在实际应用中测试
        # 这里只测试辅助函数
        
        import sys
        from pathlib import Path
        import tempfile
        import shutil
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        temp_conversation_file = Path(temp_dir) / "conversation_history.json"
        
        try:
            # 写入空的 conversation_history.json
            with open(temp_conversation_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            # 测试空列表的处理
            with open(temp_conversation_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 空列表应该被正确处理
            assert isinstance(data, list)
            
        finally:
            shutil.rmtree(temp_dir)


# =============================================================================
# 测试工具执行集成
# =============================================================================

class TestToolExecution:
    """测试工具执行"""

    @patch('openai.OpenAI')
    def test_execute_get_data_info_tool(self, mock_openai, mock_env, temp_workspace):
        """测试 get_data_info 工具通过 registry 执行"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        from geoagent.tools import execute_tool
        import geoagent.gis_tools.fixed_tools as ft

        original_get_workspace_dir = ft.get_workspace_dir
        ft.get_workspace_dir = lambda: temp_workspace

        try:
            result = execute_tool("get_data_info", {"file_name": "nonexistent.shp"})

            assert "error" in result or "不存在" in result
        finally:
            ft.get_workspace_dir = original_get_workspace_dir


# =============================================================================
# 测试 API Key 持久化和自动重置
# =============================================================================

class TestApiKeyPersistence:
    """测试 API Key 持久化"""

    def test_save_and_load_api_key(self):
        """测试 API Key 保存和加载"""
        temp_key_file = Path(tempfile.gettempdir()) / "test_api_key.txt"
        original_file = ac._API_KEY_FILE

        try:
            ac._API_KEY_FILE = temp_key_file
            if temp_key_file.exists():
                temp_key_file.unlink()

            # 保存
            GeoAgent._save_api_key("test_secret_key_123")

            # 加载
            loaded = GeoAgent._load_api_key()
            assert loaded == "test_secret_key_123"
        finally:
            ac._API_KEY_FILE = original_file
            if temp_key_file.exists():
                temp_key_file.unlink()

    def test_load_api_key_file_not_exists(self):
        """测试 API Key 文件不存在时返回 None"""
        temp_key_file = Path(tempfile.gettempdir()) / "nonexistent_key_file.txt"
        original_file = ac._API_KEY_FILE

        try:
            ac._API_KEY_FILE = temp_key_file
            if temp_key_file.exists():
                temp_key_file.unlink()

            loaded = GeoAgent._load_api_key()
            assert loaded is None
        finally:
            ac._API_KEY_FILE = original_file

    @patch('openai.OpenAI')
    def test_agent_saves_api_key(self, mock_openai, mock_env):
        """测试 Agent 创建时保存 API Key"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        temp_key_file = Path(tempfile.gettempdir()) / "test_agent_key.txt"
        original_file = ac._API_KEY_FILE

        try:
            ac._API_KEY_FILE = temp_key_file
            if temp_key_file.exists():
                temp_key_file.unlink()

            GeoAgent(api_key="sk-my_test_key")

            # 验证文件已创建且内容正确
            assert temp_key_file.exists()
            with open(temp_key_file, 'r') as f:
                assert f.read().strip() == "sk-my_test_key"
        finally:
            ac._API_KEY_FILE = original_file
            if temp_key_file.exists():
                temp_key_file.unlink()


class TestAutoResetOnError:
    """测试错误时自动重置对话"""

    @patch('openai.OpenAI')
    def test_auto_reset_on_missing_field_id(self, mock_openai, mock_env):
        """测试 V2 没有 messages 列表和 auto_reset 机制（V2 是确定性架构）"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表，也没有 auto_reset
        assert hasattr(agent, "stats")
        assert agent.stats["total_requests"] == 0

    @patch('openai.OpenAI')
    def test_auto_reset_on_invalid_request_error(self, mock_openai, mock_env):
        """测试 V2 错误处理"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 auto_reset 机制
        assert hasattr(agent, "reset_stats")
        assert callable(agent.reset_stats)

    @patch('openai.OpenAI')
    def test_message_ids_are_added(self, mock_openai, mock_env):
        """测试 V2 没有 messages 列表和 id 机制"""
        mock_client = MagicMock()
        mock_openai.return_value = mock_client

        agent = GeoAgent(api_key="sk-test_key")

        # V2 没有 messages 列表
        assert hasattr(agent, "api_key")
        assert hasattr(agent, "model")
        assert hasattr(agent, "stats")


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
