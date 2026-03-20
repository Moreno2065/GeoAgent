"""
GeoAgent Streamlit Web 应用入口
直接运行根目录的 app.py
"""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    root = __file__.rsplit("src", 1)[0]
    sys.exit(subprocess.call([sys.executable, "-m", "streamlit", "run",
                               str(Path(root) / "app.py"), "--server.headless=true"]))
