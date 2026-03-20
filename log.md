# GeoAgent 项目改动日志

## 2025-03-19 侧边栏 UI 调整

### 修改文件
- `app.py`

### 改动说明

1. **字体统一为黑色**
   - 全局变量 `--sidebar-text`、`--sidebar-text-active` 改为 `#000000`
   - 侧边栏内所有 label、.stMarkdown、p、.stCaption、input 文字颜色改为 `#000000`
   - 区块标题「历史对话」「设置」「数据文件」、副标题「空间智能分析助手」、新建对话按钮、历史项、设置项、暂无对话记录等文字均改为黑色

2. **图标放大**
   - Logo 图标：由 44×44px、font-size 24px 调整为 56×56px、font-size 32px，圆角 14px
   - 历史项对话图标：由 18px 调整为 26px，并为 💬 增加 `.history-item-icon` 样式
   - 「数据文件」前的 📂 增加 `.section-emoji`，字号 20px，与文字垂直居中

3. **排版与间距**
   - 侧边栏宽度由 280px 调整为 300px
   - 顶部 header：padding 24px 20px 20px，logo 与标题 gap 14px，标题 20px 字重 700，副标题 13px 字重 500、左对齐（padding-left 取消）
   - 新建对话按钮：padding 12px 16px，margin 20px 16px 12px，字号 14px 字重 600
   - 「历史对话」标题：13px 字重 600，padding 12px 12px 6px
   - 历史项：gap 12px，padding 10px 12px，字号 14px
   - 设置区标题：13px 字重 600，margin-bottom 10px，「数据文件」区块 margin-top 16px
   - 侧边栏底部 footer、settings 项 padding/gap 略增，字号 14px

4. **样式细节**
   - 新增 `.settings-label .section-emoji` 用于放大区块前的 emoji
   - 历史项按钮使用 `<span class='history-item-icon'>💬</span>` 以应用放大图标样式

---

## 2025-03-19 欢迎页提示框与推荐区

### 修改文件
- `app.py`

### 改动说明

1. **提示框圆角、居中、放大**
   - 欢迎页表单容器：`max-width: 720px`，`margin: 0 auto 32px`，`padding: 28px 32px`，`border-radius: 24px`，白底 + 浅边框 + 轻阴影
   - 内部 textarea：圆角 14px，最小高度 120px，字号 15px，内边距 14px 16px
   - 发送按钮：圆角 12px，字重 600
   - 列宽由 `[1, 2, 1]` 改为 `[0.5, 4, 0.5]`，中间输入区更宽

2. **删除推荐问题区块**
   - 移除「推荐问题」标题及下方 2×2 四张推荐卡片（矢量空间分析、地形坡度提取、空间连接统计、可达性分析）及对应「使用」按钮
   - 移除 `prompts` 变量与 `example_prompt` 的点击处理逻辑
