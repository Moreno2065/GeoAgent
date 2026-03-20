"""
GeoAgent CLI - 命令行入口点
"""

import sys
import os
import argparse
import subprocess


def main():
    """CLI 主入口"""
    parser = argparse.ArgumentParser(
        description="GeoAgent - 空间智能 GIS 分析 Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志"
    )
    parser.add_argument("command", nargs="?", help="子命令（version/info）")

    args = parser.parse_args()

    # 子命令模式（version / info）
    if args.command in ("version", "--version", "-V"):
        from geoagent import __version__
        print(f"GeoAgent {__version__}")
        return
    if args.command in ("info",):
        print_info()
        return
    if args.command in ("--help", "-h") or args.command is None:
        pass  # 继续启动 Web 界面

    print("GeoAgent - 空间智能 GIS 分析 Agent")
    print("=" * 50)
    print("正在启动 Streamlit Web 界面...")

    # 通过环境变量传递配置
    if args.verbose:
        os.environ["GEOAGENT_VERBOSE"] = "1"

    # 使用 subprocess 启动 Streamlit（更安全、更可控）
    cmd = [sys.executable, "-m", "streamlit", "run", "app.py", "--browser.serverAddress=localhost"]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"启动 Streamlit 失败: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n已停止 GeoAgent")
        sys.exit(0)


def print_info():
    """打印安装信息"""
    from geoagent import __version__
    import sys

    print(f"""GeoAgent 安装信息
================
版本: {__version__}
Python: {sys.version}
安装路径: {__file__}
工作目录: {os.getcwd()}

可用模块:
  - geoagent.core             : GeoAgent 类（Function Calling 模式）
  - geoagent.gis_tools       : GIS 工具
  - geoagent.plugins         : 插件系统（高德/OSM）""")


if __name__ == "__main__":
    main()
