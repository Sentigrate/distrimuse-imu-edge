from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from distrimuse_imu_edge.data.config import DataConfig
from distrimuse_imu_edge.data.sequence import SequenceWindowDataset
from distrimuse_imu_edge.data.windowing import (
    ChannelNormalizer,
    build_raw_window_dataset,
    load_window_cache,
    save_window_cache,
    stable_hash,
)


class IMUEdgeDataModule:
    """Standalone PyTorch data module for IMU edge experiments."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config
        self.normalizer = ChannelNormalizer()
        self.datasets: dict[str, SequenceWindowDataset] = {}
        self.class_weights: np.ndarray | None = None

    def setup(self) -> None:
        split_frames = self._load_split_frames()
        manifest = self._load_manifest()
        arrays: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
        for split_name, configured_ids in {
            "train": self.config.split.train_ids,
            "val": self.config.split.val_ids,
            "test": self.config.split.test_ids,
        }.items():
            # When split_dir is set, the parquet file's own row membership *is*
            # the split (e.g. synthetic subjects added to train.parquet by
            # distrimuse-synthetic-data) — trust it instead of the static
            # config.split.*_ids allowlist, which only lists real subject IDs
            # and would silently drop every synthetic row otherwise.
            participants = (
                sorted(split_frames[split_name]["person_id"].unique().tolist())
                if self.config.split_dir is not None
                else configured_ids
            )
            arrays[split_name] = self._load_or_build_windows(
                split_name=split_name,
                df=split_frames[split_name],
                participants=participants,
                manifest=manifest,
            )

        x_train, y_train, *_ = arrays["train"]
        self.normalizer.fit(x_train)
        self.class_weights = self._compute_class_weights(y_train)

        for split_name, (x, y, pids, sids, starts) in arrays.items():
            x_norm = self.normalizer.transform(x) if len(x) else x
            self.datasets[split_name] = SequenceWindowDataset(
                x_norm,
                y,
                pids,
                sids,
                context_len=self.config.context_len,
                future_context_len=self.config.future_context_len,
                window_starts_s=starts,
            )

    def _load_manifest(self) -> pd.DataFrame | None:
        if self.config.manifest_path is None:
            return None
        return pd.read_parquet(self.config.manifest_path)

    def _load_split_frames(self) -> dict[str, pd.DataFrame]:
        if self.config.split_dir is not None:
            return {
                split: pd.read_parquet(self.config.split_dir / f"{split}.parquet")
                for split in ("train", "val", "test")
            }
        all_frames = self._load_processed_frames(self.config.split.person_ids)
        return {
            "train": all_frames[all_frames["person_id"].isin(self.config.split.train_ids)].copy(),
            "val": all_frames[all_frames["person_id"].isin(self.config.split.val_ids)].copy(),
            "test": all_frames[all_frames["person_id"].isin(self.config.split.test_ids)].copy(),
        }

    def _load_processed_frames(self, person_ids: list[int]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for person_id in person_ids:
            for scenario_id in self.config.split.scenario_ids:
                path = self._processed_local_path(person_id, scenario_id)
                if path.exists():
                    frame = pd.read_parquet(path)
                else:
                    frame = self._download_processed_session(person_id, scenario_id)
                frame = frame.copy()
                frame["person_id"] = person_id
                frame["scenario_id"] = scenario_id
                frames.append(frame)
        if not frames:
            raise FileNotFoundError(
                "No processed IMU frames found. Set data.split_dir or data.processing_version/cache_dir."
            )
        return pd.concat(frames, ignore_index=True)

    def _processed_local_path(self, person_id: int, scenario_id: int) -> Path:
        if self.config.processing_version is None:
            return Path("__missing__")
        return (
            self.config.cache_dir
            / "processed"
            / self.config.campaign
            / "imu"
            / self.config.processing_version
            / f"person_{person_id}"
            / f"scenario_{scenario_id}"
            / "data.parquet"
        )

    def _download_processed_session(self, person_id: int, scenario_id: int) -> pd.DataFrame:
        if self.config.processing_version is None:
            raise FileNotFoundError("data.processing_version is required when data.split_dir is not set")
        from distrimuse_core.io import FusionDataStore

        store = FusionDataStore(cache_dir=self.config.cache_dir)
        key = (
            f"processed/{self.config.campaign}/imu/{self.config.processing_version}/"
            f"person_{person_id}/scenario_{scenario_id}/data.parquet"
        )
        local_path = store.download(key)
        return pd.read_parquet(local_path)

    def _window_cache_path(self, *, split_name: str, participants: list[int]) -> Path:
        key = stable_hash(
            {
                "split": split_name,
                "participants": participants,
                "campaign": self.config.campaign,
                "processing_version": self.config.processing_version,
                "window_size_s": self.config.window_size_s,
                "hop_size_s": self.config.hop_size_s,
                "sensor_cols": self.config.sensor_cols,
                "task_col": self.config.task_col,
                "manifest": self.config.manifest_path,
            }
        )
        return self.config.window_cache_dir / f"{split_name}_{key}.npz"

    def _load_or_build_windows(
        self,
        *,
        split_name: str,
        df: pd.DataFrame,
        participants: list[int],
        manifest: pd.DataFrame | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        cache_path = self._window_cache_path(split_name=split_name, participants=participants)
        if self.config.reuse_window_cache and cache_path.exists():
            return load_window_cache(cache_path)
        arrays = build_raw_window_dataset(
            df,
            participants,
            task_col=self.config.task_col,
            window_size_s=self.config.window_size_s,
            hop_size_s=self.config.hop_size_s,
            sensor_cols=self.config.sensor_cols,
            window_manifest=manifest,
            fs=self.config.fs,
        )
        save_window_cache(
            cache_path, x=arrays[0], y=arrays[1], pids=arrays[2], sids=arrays[3], starts=arrays[4]
        )
        return arrays

    def train_loader(self) -> DataLoader:
        return self._loader("train", shuffle=True)

    def val_loader(self) -> DataLoader:
        return self._loader("val", shuffle=False)

    def test_loader(self) -> DataLoader:
        return self._loader("test", shuffle=False)

    def _loader(self, split_name: str, *, shuffle: bool) -> DataLoader:
        return DataLoader(
            self.datasets[split_name],
            batch_size=self.config.batch_size,
            shuffle=shuffle,
            num_workers=self.config.num_workers,
            pin_memory=False,
        )

    def split_metadata(self, split_name: str) -> dict[str, np.ndarray]:
        return self.datasets[split_name].metadata()

    def _compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        counts = np.bincount(y.astype(np.int64), minlength=self.config.n_classes).astype(np.float32)
        weights = np.ones(self.config.n_classes, dtype=np.float32)
        present = counts > 0
        if present.any():
            weights[present] = counts[present].sum() / counts[present]
            weights[present] /= weights[present].mean()
        return weights
