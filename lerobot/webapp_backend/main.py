from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request, send_from_directory

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


LOCAL_DATASETS: list[dict[str, Any]] = []


app = Flask(__name__, static_folder=None)


def _find_dataset_config(repo_id: str) -> dict[str, Any] | None:
    for cfg in LOCAL_DATASETS:
        if cfg.get("repo_id") == repo_id:
            return cfg
    return None


@lru_cache(maxsize=16)
def get_dataset(repo_id: str) -> LeRobotDataset:
    cfg = _find_dataset_config(repo_id)
    if cfg is None:
        raise KeyError(f"Unknown dataset repo_id: {repo_id}")
    root = Path(cfg["root"])
    return LeRobotDataset(repo_id=repo_id, root=root, tolerance_s=cfg.get("tolerance_s", 1e-4))


@app.get("/api/health")
def health() -> Any:
    return jsonify({"status": "ok"})


@app.get("/api/datasets")
def list_datasets() -> Any:
    items: list[dict[str, Any]] = []
    for cfg in LOCAL_DATASETS:
        repo_id = cfg["repo_id"]
        dataset = get_dataset(repo_id)
        fps = dataset.fps
        num_frames = dataset.num_frames
        num_episodes = dataset.num_episodes
        total_hours = float(num_frames) / float(fps) / 3600.0 if fps > 0 else 0.0
        tasks_set: set[str] = set()
        for episode in dataset.meta.episodes.values():
            for task in episode.get("tasks", []):
                tasks_set.add(task)
        items.append(
            {
                "repo_id": repo_id,
                "root": str(cfg["root"]),
                "num_episodes": num_episodes,
                "num_frames": num_frames,
                "fps": fps,
                "total_hours": total_hours,
                "tasks": sorted(tasks_set),
            }
        )
    return jsonify(items)


@app.get("/api/datasets/<path:repo_id>/episodes")
def list_episodes(repo_id: str) -> Any:
    try:
        dataset = get_dataset(repo_id)
    except KeyError:
        return jsonify({"error": "dataset_not_found"}), 404
    episodes: list[dict[str, Any]] = []
    for episode_index, info in dataset.meta.episodes.items():
        length = int(info.get("length", 0))
        duration_seconds = float(length) / float(dataset.fps) if dataset.fps > 0 else 0.0
        episodes.append(
            {
                "episode_index": episode_index,
                "length": length,
                "duration_seconds": duration_seconds,
                "tasks": info.get("tasks", []),
            }
        )
    episodes.sort(key=lambda x: x["episode_index"])
    return jsonify(episodes)


@app.get("/api/datasets/<path:repo_id>/episodes/<int:episode_index>")
def episode_detail(repo_id: str, episode_index: int) -> Any:
    try:
        dataset = get_dataset(repo_id)
    except KeyError:
        return jsonify({"error": "dataset_not_found"}), 404
    info = dataset.meta.episodes.get(episode_index)
    if info is None:
        return jsonify({"error": "episode_not_found"}), 404
    length = int(info.get("length", 0))
    duration_seconds = float(length) / float(dataset.fps) if dataset.fps > 0 else 0.0
    has_action = "action" in dataset.features
    has_state = "observation.state" in dataset.features
    return jsonify(
        {
            "episode_index": episode_index,
            "length": length,
            "duration_seconds": duration_seconds,
            "tasks": info.get("tasks", []),
            "fps": dataset.fps,
            "has_action": has_action,
            "has_state": has_state,
        }
    )


@app.get("/api/datasets/<path:repo_id>/episodes/<int:episode_index>/playback")
def episode_playback(repo_id: str, episode_index: int) -> Any:
    try:
        dataset = get_dataset(repo_id)
    except KeyError:
        return jsonify({"error": "dataset_not_found"}), 404
    if episode_index not in dataset.meta.episodes:
        return jsonify({"error": "episode_not_found"}), 404
    from_index = int(dataset.episode_data_index["from"][episode_index].item())
    to_index = int(dataset.episode_data_index["to"][episode_index].item())
    indices = list(range(from_index, to_index))
    hf_dataset = dataset.hf_dataset.select(indices)
    timestamps_raw = hf_dataset["timestamp"]
    timestamps = [float(t) for t in timestamps_raw]
    selected_columns: list[str] = []
    for col, ft in dataset.features.items():
        if ft.get("dtype") in ["float32", "int32"] and col != "timestamp":
            shape = ft.get("shape")
            if isinstance(shape, (list, tuple)) and len(shape) == 1:
                selected_columns.append(col)
    series: list[dict[str, Any]] = []
    for column_name in selected_columns:
        feature_info = dataset.features[column_name]
        dim = int(feature_info["shape"][0])
        names = feature_info.get("names")
        if names:
            column_names = names
            while not isinstance(column_names, list):
                column_names = list(column_names.values())[0]
        else:
            column_names = [f"{column_name}_{i}" for i in range(dim)]
        values_raw = hf_dataset[column_name]
        values = [v.tolist() for v in values_raw]
        series.append(
            {
                "key": column_name,
                "names": column_names,
                "values": values,
            }
        )
    return jsonify(
        {
            "fps": dataset.fps,
            "timestamps": timestamps,
            "series": series,
        }
    )


@app.get("/api/datasets/<path:repo_id>/episodes/<int:episode_index>/videos")
def episode_videos(repo_id: str, episode_index: int) -> Any:
    try:
        dataset = get_dataset(repo_id)
    except KeyError:
        return jsonify({"error": "dataset_not_found"}), 404
    if episode_index not in dataset.meta.episodes:
        return jsonify({"error": "episode_not_found"}), 404
    videos: list[dict[str, Any]] = []
    for key in dataset.meta.video_keys:
        video_path = dataset.root / dataset.meta.get_video_file_path(episode_index, key)
        videos.append(
            {
                "camera_key": key,
                "path": str(video_path),
            }
        )
    return jsonify(videos)


@app.route("/")
def index() -> Any:
    frontend_dir = Path(__file__).resolve().parent.parent / "webapp_frontend"
    return send_from_directory(frontend_dir, "index.html")


@app.route("/frontend/<path:filename>")
def frontend_files(filename: str) -> Any:
    frontend_dir = Path(__file__).resolve().parent.parent / "webapp_frontend"
    return send_from_directory(frontend_dir, filename)


def create_app() -> Flask:
    return app
