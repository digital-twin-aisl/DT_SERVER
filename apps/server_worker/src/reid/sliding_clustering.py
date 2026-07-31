from collections import deque
from dataclasses import dataclass, field
import json
import os
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import DBSCAN


@dataclass
class TrackState:
    """DBSCAN window와 별개로 유지되는 Re-ID track 상태."""

    track_id: int
    centroid: np.ndarray
    last_seen_step: int
    global_id: int | None = None
    consecutive_hits: int = 1
    gallery: deque[np.ndarray] = field(default_factory=deque)


class ClusteringSliding:
    """Sliding-window clustering과 persistent Global ID tracking을 수행한다.

    DBSCAN은 최근 window에서 현재 관측이 속한 appearance cluster를 구한다.
    Global ID는 별도의 track 상태로 max_age 동안 유지하며, gallery 기반
    Hungarian matching으로 cluster와 연결한다. 새 cluster는 연속 관측으로
    확인된 뒤에만 Global ID를 발급한다.
    """

    def __init__(
        self,
        edges,
        window_size=10,
        data=None,
        *,
        max_age=30,
        confirmation_hits=3,
        match_similarity=0.8,
        gallery_size=30,
    ):
        self.data = data
        self.edges = edges
        self.window_size = int(window_size)
        self.max_age = int(max_age)
        self.confirmation_hits = int(confirmation_hits)
        self.match_similarity = float(match_similarity)
        self.gallery_size = int(gallery_size)
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        if self.max_age < 1:
            raise ValueError("max_age must be positive")
        if self.confirmation_hits < 1:
            raise ValueError("confirmation_hits must be positive")
        if not 0.0 <= self.match_similarity <= 1.0:
            raise ValueError("match_similarity must be between 0 and 1")
        if self.gallery_size < 1:
            raise ValueError("gallery_size must be positive")

        self.frame_counter = 0
        self.frame_buffer = deque(maxlen=self.window_size)

        self._cluster_step = 0
        self._next_track_id = 0
        self._tracks: dict[int, TrackState] = {}

        # 기존 코드/진단 도구와의 호환성을 위해 confirmed track만 노출한다.
        self.previous_centroids: dict[int, np.ndarray] = {}
        self.next_global_id = 0

    @staticmethod
    def _metadata(item: dict[str, Any]) -> dict[str, Any]:
        metadata = {
            "edge_id": str(item["edge_id"]),
            "cam": int(item["cam"]),
            "frame": int(item["frame"]),
            "bbox": item["bbox"],
            "feature": item["feature"],
        }
        if "person_idx" in item:
            metadata["person_idx"] = int(item["person_idx"])
        return metadata

    def _append_frame(self, data: list[dict[str, Any]] | None) -> None:
        frame_features = []
        frame_metadata = []
        for item in data or []:
            if item.get("feature") is None:
                continue
            frame_features.append(np.asarray(item["feature"], dtype=np.float32))
            frame_metadata.append(self._metadata(item))
        self.frame_buffer.append((frame_features, frame_metadata))
        self.frame_counter += 1

    def process_json(self, json_path):
        with open(json_path, encoding="utf-8") as input_file:
            data = json.load(input_file)
        self._append_frame(data)
        if len(self.frame_buffer) >= self.window_size:
            return self.cluster_and_save()
        return []

    def process_realtime(self, data):
        self._append_frame(data)
        if len(self.frame_buffer) >= self.window_size:
            # Root association에는 현재 synchronized batch의 bbox만 전달한다.
            return self.cluster(latest_only=True)
        return []

    @staticmethod
    def _normalize(feature: np.ndarray) -> np.ndarray | None:
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        if not np.isfinite(feature).all():
            return None
        norm = float(np.linalg.norm(feature))
        if norm <= 1e-12:
            return None
        return feature / norm

    def _expire_tracks(self) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if self._cluster_step - track.last_seen_step > self.max_age
        ]
        for track_id in expired:
            del self._tracks[track_id]

    def _sync_previous_centroids(self) -> None:
        self.previous_centroids = {
            track.global_id: track.centroid
            for track in self._tracks.values()
            if track.global_id is not None
        }

    @staticmethod
    def _gallery_similarity(track: TrackState, centroid: np.ndarray) -> float:
        references = list(track.gallery) or [track.centroid]
        return max(float(np.dot(reference, centroid)) for reference in references)

    def _update_track(self, track: TrackState, centroid: np.ndarray) -> None:
        was_consecutive = track.last_seen_step == self._cluster_step - 1
        track.consecutive_hits = track.consecutive_hits + 1 if was_consecutive else 1

        updated_centroid = self._normalize(0.8 * track.centroid + 0.2 * centroid)
        track.centroid = centroid if updated_centroid is None else updated_centroid
        track.last_seen_step = self._cluster_step
        track.gallery.append(centroid)
        while len(track.gallery) > self.gallery_size:
            track.gallery.popleft()

        if (
            track.global_id is None
            and track.consecutive_hits >= self.confirmation_hits
        ):
            track.global_id = self.next_global_id
            self.next_global_id += 1

    def _new_track(self, centroid: np.ndarray) -> TrackState:
        track = TrackState(
            track_id=self._next_track_id,
            centroid=centroid,
            last_seen_step=self._cluster_step,
            gallery=deque([centroid]),
        )
        self._next_track_id += 1
        if self.confirmation_hits == 1:
            track.global_id = self.next_global_id
            self.next_global_id += 1
        self._tracks[track.track_id] = track
        return track

    def _match_tracks(
        self,
        current_centroids: dict[int, np.ndarray],
    ) -> dict[int, int]:
        self._cluster_step += 1
        self._expire_tracks()

        labels = sorted(current_centroids)
        tracks = list(self._tracks.values())
        label_to_track: dict[int, TrackState] = {}

        if labels and tracks:
            similarities = np.asarray(
                [
                    [
                        self._gallery_similarity(track, current_centroids[label])
                        for track in tracks
                    ]
                    for label in labels
                ],
                dtype=np.float64,
            )
            valid = similarities >= self.match_similarity
            costs = np.where(valid, 1.0 - similarities, 1e6)
            rows, columns = linear_sum_assignment(costs)
            for row, column in zip(rows, columns):
                if not valid[row, column]:
                    continue
                label = labels[row]
                track = tracks[column]
                self._update_track(track, current_centroids[label])
                label_to_track[label] = track

        for label in labels:
            if label not in label_to_track:
                label_to_track[label] = self._new_track(current_centroids[label])

        self._sync_previous_centroids()
        return {
            label: track.global_id
            for label, track in label_to_track.items()
            if track.global_id is not None
        }

    def cluster(self, *, latest_only=False) -> list[dict[str, Any]]:
        features = []
        metadata = []
        is_latest = []
        feature_size = None
        latest_buffer_index = len(self.frame_buffer) - 1

        for buffer_index, (frame_features, frame_metadata) in enumerate(
            self.frame_buffer
        ):
            for feature, item in zip(frame_features, frame_metadata):
                normalized = self._normalize(feature)
                if normalized is None:
                    continue
                if feature_size is None:
                    feature_size = len(normalized)
                if len(normalized) != feature_size:
                    continue
                features.append(normalized)
                metadata.append(item)
                is_latest.append(buffer_index == latest_buffer_index)

        if not features:
            self._cluster_step += 1
            self._expire_tracks()
            self._sync_previous_centroids()
            return []

        feature_array = np.stack(features)
        labels = DBSCAN(
            eps=0.05,
            min_samples=5,
            metric="cosine",
        ).fit_predict(feature_array)

        cluster_to_indices: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            if label != -1:
                cluster_to_indices.setdefault(int(label), []).append(index)

        # 과거 window에만 남아 있는 cluster는 track의 last_seen을 갱신하지 않는다.
        latest_labels = {
            int(label)
            for index, label in enumerate(labels)
            if is_latest[index] and label != -1
        }
        current_centroids = {}
        for label, indices in cluster_to_indices.items():
            if label not in latest_labels:
                continue
            centroid = self._normalize(np.mean(feature_array[indices], axis=0))
            if centroid is not None:
                current_centroids[label] = centroid

        label_to_global_id = self._match_tracks(current_centroids)

        results = []
        for index, item in enumerate(metadata):
            if latest_only and not is_latest[index]:
                continue
            label = int(labels[index])
            if label == -1 or label not in label_to_global_id:
                continue
            results.append(
                {
                    "global_id": label_to_global_id[label],
                    **item,
                }
            )
        return results

    def cluster_and_save(self):
        results = self.cluster(latest_only=False)
        result_dir = getattr(self, "result_dir", None)
        if result_dir is not None:
            end_frame = self.frame_counter - 1
            json_path = os.path.join(result_dir, f"frame_{end_frame:05d}.json")
            with open(json_path, "w", encoding="utf-8") as output_file:
                json.dump(results, output_file, indent=2)
        return results

    def finalize(self):
        print("모든 프레임 데이터 클러스터링 완료.")
