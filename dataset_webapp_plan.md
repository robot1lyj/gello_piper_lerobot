# 本地 LeRobot 数据集管理 Web 应用规划

> 目标：在本地纯内网环境下，构建一个基于 **Flask + Vue3 + ECharts** 的应用级数据集管理与审核系统，仅依赖本地 `LeRobotDataset`，不依赖 Hugging Face 在线服务。

---

## 1. 背景与总体目标

- **背景**
  - 当前已有官方脚本：`visualize_dataset.py`、`visualize_dataset_html.py`，可以做到单 episode 的可视化，但偏“工具脚本”，不适合作为长期的数据管理平台。
  - 项目要求数据仅存本地，禁止依赖 HF Hub 网络访问，系统部署范围限定在内网中。

- **总体目标**
  - 建立一个 **美观、稳定、可扩展** 的数据集管理 Web 应用，满足：
    - 多数据集统一管理
    - 多相机视角 + 动作/状态曲线的回放
    - 数据统计与仪表盘
    - 数据筛选（放弃/保留）与审核流
  - 同时作为学习项目，采用 **Vue3 前端**，逐步掌握实际工程中的前后端开发模式。

---

## 2. 本地数据目录与数据集组织

### 2.1 顶层目录设计

推荐在每台机器上约定一个根目录，例如：

- Ubuntu：`/data/lerobot_datasets`
- Windows：`D:/lerobot_datasets`

目录结构示例：

```text
/data/lerobot_datasets/
  project_x/                       # 可选：按项目或机器人分组
    lerobot_pusht/                 # repo_id = "lerobot/pusht"
      meta/
        info.json
        tasks.jsonl
        episodes.jsonl
        stats.json
        review.jsonl               # ✅ 新增：审核与筛选元数据
      data/
        episode_000.parquet
        episode_001.parquet
        ...
      videos/
        episode_000_front.mp4
        episode_000_wrist.mp4
        ...
      logs/                        # 可选：训练或导出日志
      cache/                       # 可选：中间结果
    lerobot_aloha/
      ...
  project_y/
    ...
```

### 2.2 数据集注册表（Registry）

为避免扫描整块磁盘，后端通过一个“数据集注册表”来管理可用数据集：

- 目前在 `lerobot/webapp_backend/main.py` 中使用 Python 列表 `LOCAL_DATASETS`：

```python
LOCAL_DATASETS = [
    {
        "repo_id": "lerobot/pusht",
        "root": "/data/lerobot_datasets/project_x/lerobot_pusht",
        "tolerance_s": 1e-4,
        "tags": ["pusht", "sim", "v2.1"],
    },
    {
        "repo_id": "lerobot/aloha_static_cups_open",
        "root": "/data/lerobot_datasets/project_x/lerobot_aloha",
        "tolerance_s": 1e-4,
        "tags": ["aloha", "real", "cups"],
    },
]
```

- 后续可选优化：将 registry 抽到独立 JSON/YAML 文件，Flask 启动时加载，方便在不改代码的情况下增加/删除数据集。

### 2.3 大量数据集时的策略

当数据集数量较多（几十上百）时：

- **分层组织**：按项目/机器人/场景划分子目录，避免单目录过长。
- **标签化管理**：在 registry 中为每个数据集添加 `tags` 字段（如 `"sim"` / `"real"` / `"push"` / `"pick"`）。
- **前端分页与搜索**：
  - `/datasets` 页面默认分页显示数据集列表。
  - 支持按名称、repo_id、tags 搜索。
- **按需加载 episodes**：
  - `/api/datasets` 只返回数据集级别信息。
  - 请求 `/api/datasets/{repo_id}/episodes` 时才查询 episode 列表，减少一次性加载量。

---

## 3. 后端架构与 API 规划

### 3.1 架构概览

- 技术栈：
  - Python 3.10 + Flask
  - 仅使用本地 `LeRobotDataset`，不访问 HF Hub
- 目录结构（已创建/规划）：

```text
lerobot/
  webapp_backend/
    main.py              # Flask app + API 路由（当前已存在）
    registry.py          # (规划) 数据集注册表与加载工具
    review_store.py      # (规划) 审核状态读写工具
    stats.py             # (规划) 统计逻辑
```

### 3.2 已有 API（初版）

在 `webapp_backend/main.py` 中：

- `GET /api/health`：健康检查。
- `GET /api/datasets`：列出所有 registry 中的本地数据集（含帧数、时长、任务列表）。
- `GET /api/datasets/<repo_id>/episodes`：返回该数据集所有 episodes 的简要信息。
- `GET /api/datasets/<repo_id>/episodes/<episode_index>`：单 episode 元数据（时长、任务、是否有动作/状态）。
- `GET /api/datasets/<repo_id>/episodes/<episode_index>/playback`：episode 的时间序列数据，供前端绘制曲线。
- `GET /api/datasets/<repo_id>/episodes/<episode_index>/videos`：episode 对应的视频文件路径（按相机分组）。

### 3.3 规划新增 API

#### 3.3.1 统计相关

- `GET /api/stats/datasets`
  - 返回每个数据集的：
    - `num_episodes`, `num_frames`, `total_hours`, `tasks`, 审核状态统计。
- `GET /api/stats/tasks`
  - 跨数据集按任务聚合：
    - 每个任务的总 episodes、总时长、涉及数据集数。
- `GET /api/stats/review`
  - 全局审核状态统计：不同状态（pending/approved/rejected/flagged）的 episodes 总数与占比。

#### 3.3.2 审核与筛选相关

引入 `meta/review.jsonl` 存储审核信息（详见第 5 节）：

- `GET /api/datasets/<repo_id>/review`
  - 返回该数据集所有 episodes 的审核状态映射：

    ```json
    {
      "0": {"status": "approved", "tags": ["good"], ...},
      "1": {"status": "rejected", "tags": ["camera_issue"], ...}
    }
    ```

- `POST /api/datasets/<repo_id>/episodes/<episode_index>/review`
  - 设置单个 episode 的审核信息：

    ```json
    {
      "status": "rejected",            // pending / approved / rejected / flagged
      "tags": ["camera_issue"],
      "comment": "相机抖动严重",
      "reviewer": "alice"
    }
    ```

- （可选）`POST /api/datasets/<repo_id>/episodes/batch_review`
  - 批量更新多条 episodes 的状态，便于大规模数据筛选。

---

## 4. 前端架构与页面规划（Vue3 + ECharts）

### 4.1 技术栈

- 使用 Vue3（CDN 版本，已在 `webapp_frontend/index.html` 初步接入）。
- 图表库：ECharts（CDN）。
- 当前为单文件 SPA，后续可按需要引入 Vue Router / 组件化拆分。

### 4.2 页面设计

#### 4.2.1 数据集总览页 `/datasets`

- 功能：
  - 展示所有数据集（表格 / 卡片）。
  - 支持搜索（名称 / repo_id）与标签筛选。
  - 显示基础指标：episodes、总时长、主要任务、审核进度。
- 数据来源：
  - `GET /api/datasets`
  - `GET /api/stats/datasets`

#### 4.2.2 数据集详情页 `/datasets/:repo_id`

- 功能：
  - 展示单个数据集的基本属性（fps、帧数、总时长、任务列表）。
  - 任务统计：按任务的 episodes 数与时长（ECharts 条形图）。
  - 审核统计：pending/approved/rejected/flagged 数量和比例。
  - episodes 表格：
    - 列：episode_index、时长、任务、审核状态、标签、操作（回放 / 审核）。
    - 支持筛选（任务 / 审核状态 / 时长区间），排序（时长 / index）。
- 数据来源：
  - `GET /api/datasets/<repo_id>/episodes`
  - `GET /api/datasets/<repo_id>/review`

#### 4.2.3 Episode 回放页 `/datasets/:repo_id/episodes/:episode_index`

- 功能：
  - 已有的多相机视频 + 曲线回放能力。
  - 增加审核操作区：
    - 状态按钮：通过（approved）、拒绝（rejected）、标记（flagged）。
    - 备注输入框：comment。
    - tag 选择/输入框：tags（如“camera_issue”、“bad_label”等）。
  - 点击保存后调用对应的 `POST /review` 接口，更新 `review.jsonl`。
- 数据来源：
  - `GET /api/datasets/<repo_id>/episodes/<episode_index>`
  - `GET /api/datasets/<repo_id>/episodes/<episode_index>/videos`
  - `GET /api/datasets/<repo_id>/episodes/<episode_index>/playback`
  - `GET /api/datasets/<repo_id>/review`（获取当前状态）

#### 4.2.4 统计仪表盘 `/stats`

- 功能：
  - 全局核心统计：
    - 总数据集数、总 episodes、总时长（小时）。
  - 任务分布：任务名称 vs 总时长 / episodes 数（柱状图）。
  - 审核状态分布：pie + 堆叠条形图（按数据集拆分）。
- 数据来源：
  - `GET /api/stats/datasets`
  - `GET /api/stats/tasks`
  - `GET /api/stats/review`

#### 4.2.5 审核队列页 `/review`

- 功能：
  - 集中显示所有 `pending` 状态的 episodes。
  - 筛选条件：数据集、任务、时长范围等。
  - 一键跳转到回放页进行审核。
- 数据来源：
  - `GET /api/datasets` + `GET /api/datasets/<repo_id>/review`
  - 或设计专门的 `GET /api/review_queue` 接口返回合并结果。

---

## 5. 数据筛选 / 放弃策略（基于 LeRobot 格式）

### 5.1 Episode 状态定义

为每个 episode 增加一个逻辑状态（不改原 parquet/视频）：

- `pending`：默认状态，尚未审核。
- `approved`：已通过审核，可用于训练与下游任务。
- `rejected`：已明确放弃，不应参与训练。
- `flagged`：存在问题但需要进一步人工处理（如标注缺失）。

附加字段：

- `tags`: `list[str]` — 描述问题或特点（`"camera_issue"`, `"bad_label"`, `"partial_motion"` 等）。
- `comment`: `str` — 审核备注。
- `reviewer`: `str` — 审核人。
- `review_time`: `str` — 审核时间（ISO8601）。

### 5.2 `meta/review.jsonl` 结构

设计一个与 `episodes.jsonl` 并行的审核文件：`meta/review.jsonl`，每行代表一个 episode：

```jsonl
{"episode_index": 0, "status": "approved", "tags": ["good"], "comment": "动作顺畅", "reviewer": "alice", "review_time": "2025-12-18T10:00:00"}
{"episode_index": 1, "status": "rejected", "tags": ["camera_issue"], "comment": "相机抖动严重", "reviewer": "bob", "review_time": "2025-12-18T11:00:00"}
{"episode_index": 2, "status": "pending"}
```

特点：

- **不修改 LeRobot 原生的 meta 文件**（`info.json`、`episodes.jsonl` 等），兼容性好。
- 后端可通过一个工具类 `ReviewStore` 读写该文件，抽象为 `episode_index -> review_info` 的字典。
- 前端所有审核/筛选操作最终都体现在此文件的内容变化上。

### 5.3 训练与使用侧的接入点

训练脚本推荐调整为：

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from my_project.review import load_approved_episodes

all_dataset = LeRobotDataset(repo_id, root=root, tolerance_s=1e-4)
approved_eps = load_approved_episodes(repo_id, root)  # 从 review.jsonl 读取 status == "approved" 的 episode_index

train_dataset = LeRobotDataset(
    repo_id,
    root=root,
    episodes=approved_eps,
    tolerance_s=1e-4,
)
```

这样，“放弃数据”完全通过状态过滤实现，不需要删除/移动底层 parquet 或视频文件。

---

## 6. 数据审核制度设计

### 6.1 角色与流程

- 角色：
  - **Recorder**：数据采集者。
  - **Reviewer**：数据审核者，负责质量控制。
  - **Trainer**：模型训练负责者。

- 流程：
  1. 采集与转换：将原始数据转换为 `LeRobotDataset` 格式（已有工具承担这一部分）。
  2. 注册数据集：在 `LOCAL_DATASETS` 或 registry 文件中配置新的数据集。
  3. 待审核列表：审核者通过 `/review` 页面查看 `pending` episodes。
  4. 回放审核：在回放页面查看多相机视频与动作/状态曲线，判断质量：
     - 通过 → 标记为 `approved`。
     - 严重问题 → 标记为 `rejected` 并记录原因。
     - 可用但存在问题 → 标记为 `flagged` 并加标签。
  5. 更新元数据：审核操作通过 `POST /review` 更新 `meta/review.jsonl`。
  6. 训练/使用：训练脚本仅选取 `approved` episodes，或配置允许包含 `flagged`。

### 6.2 前端呈现方式

- 数据集详情页：
  - episodes 表格中显示彩色状态标签（pending/approved/rejected/flagged）。
  - 支持按状态筛选，用于快速定位问题数据。

- 审核队列页：
  - 默认仅显示 `pending` 状态。
  - 可按数据集/任务/时长筛选。
  - 支持批量操作（如批量标记为 `rejected`）。

- 统计页：
  - 全局审核完成率。
  - 不同状态的占比，按数据集分组展示。

---

## 7. 实施计划：一步步落地

### Phase 0（已做 & 需要补完的配置）

- [x] 创建 Flask 后端骨架：`lerobot/webapp_backend/main.py`。
- [x] 创建 Vue3 + ECharts 的前端初版：`lerobot/webapp_frontend/index.html`。
- [ ] 在 `LOCAL_DATASETS` 中填写实际本地数据集路径与 repo_id。
- [ ] 本地启动 Flask + 前端，完成端到端验证（能看到数据集列表、episode 列表、曲线）。

### Phase 1：数据注册与基础统计

- [ ] 抽取 registry 逻辑：将 `LOCAL_DATASETS` 抽为单独模块 `registry.py`，提供：
  - `list_registered_datasets()`
  - `get_dataset_config(repo_id)`
- [ ] 在后端增加 `stats.py`：
  - 计算每个数据集的 `num_episodes`、`num_frames`、`total_hours`。
  - 根据 `episodes.jsonl` 统计每个数据集的任务分布。
- [ ] 暴露 `GET /api/stats/datasets` 和 `GET /api/stats/tasks`。
- [ ] 前端 `/datasets` 页面接入统计信息，展示更丰富的摘要信息。

### Phase 2：多页面前端结构

- [ ] 在 `index.html` 中为 Vue 应用增加简单的“路由”状态（不一定引入 Vue Router）：
  - `currentView`：`"datasets" | "datasetDetail" | "playback" | "stats" | "review"`。
- [ ] 实现 `/datasets` 总览视图：
  - 调用 `/api/datasets` + `/api/stats/datasets`。
  - 支持搜索和简单筛选。
- [ ] 实现 `/datasets/:repo_id` 详情视图：
  - 调用 `/api/datasets/{repo_id}/episodes`。
  - 初步展示 episodes 表格。

### Phase 3：审核元数据与审核 API

- [ ] 在后端新增 `review_store.py`：
  - 提供读取/写入 `meta/review.jsonl` 的工具：
    - `load_review_map(root) -> dict[int, ReviewInfo]`
    - `upsert_review(root, episode_index, review_info)`。
- [ ] 实现审核相关 API：
  - `GET /api/datasets/<repo_id>/review`。
  - `POST /api/datasets/<repo_id>/episodes/<episode_index>/review`。
- [ ] 在数据集详情页中：
  - 合并 `/episodes` + `/review` 的信息，在表格中显示审核状态与标签。
- [ ] 在回放页中：
  - 显示当前 episode 的审核状态。
  - 增加审核操作表单（状态 + tags + comment）并调用 `POST /review` 保存。

### Phase 4：审核队列与统计页面

- [ ] 增加审核队列 API：
  - `GET /api/review_queue?status=pending&dataset=...&task=...`（可选，或由前端组装）。
- [ ] 实现 `/review` 页面：
  - 展示所有待审核 episodes。
  - 支持筛选和跳转到回放页。
- [ ] 实现 `/stats` 统计页：
  - 使用 ECharts 展示任务分布与审核状态统计。

### Phase 5：训练流程接入筛选逻辑

- [ ] 在项目中新增一个通用工具函数（例如 `lerobot/common/utils/review_utils.py`）：

  ```python
  def load_approved_episodes(repo_id: str, root: Path) -> list[int]:
      """读取 meta/review.jsonl，返回 status == 'approved' 的 episode_index 列表。"""
  ```

- [ ] 修改训练脚本（或新增示例脚本）：
  - 使用 `load_approved_episodes` 作为 `episodes` 参数，只训练已通过审核的数据。
- [ ] 在文档或 README 中说明：
  - “如何只使用通过审核的数据进行训练”。

---

## 8. 后续可选优化方向

- 引入 **Vue Router + 组件拆分**，让前端项目结构更清晰，便于长期维护与学习。
- 在后端增加 **分页与缓存**：应对超大规模数据集，避免一次性加载所有 episodes。
- 将审核记录同步到一个轻量数据库（如 SQLite），支持更复杂的历史追踪与统计。
- 在统计页面中增加 **时间维度** 分析（若 meta 中包含录制时间），用于观察数据分布随时间变化的趋势。
