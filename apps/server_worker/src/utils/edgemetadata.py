import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

from apps.server_worker.src.utils.transforms import get_affine_transform, get_scale


DEFAULT_METADATA_PATH = Path(__file__).resolve().parents[2] / "data" / "edge.json"


@dataclass(slots=True)
class EdgeMetadata:
    id: str
    topic: str
    cams: list[dict[str, Any]]
    transform: np.ndarray


class EdgeMetadataLoader:
    """edge.json을 읽어 모델에서 사용하는 Edge 정보로 변환한다."""

    def __init__(
        self,
        cfg: Any,
        edge_ids: list[str],
        path: str | Path = DEFAULT_METADATA_PATH,
    ) -> None:
        with Path(path).open(encoding="utf-8") as file:
            data = json.load(file)

        profiles = data["camera_profiles"]
        translation_scale = 1000 if data["translation_unit"] == "meter" else 1
        raw_edges = {edge["id"]: edge for edge in data["edges"]}
        self._edges = {}

        for edge_id in edge_ids:
            edge = raw_edges[edge_id]
            cams = [
                self._make_camera(camera, profiles, translation_scale)
                for camera in edge["cameras"]
            ]
            resolution = cams[0]["resolution"]
            self._edges[edge_id] = EdgeMetadata(
                id=edge_id,
                topic=edge["topic"],
                cams=cams,
                transform=self._make_transform(cfg, resolution),
            )

    @property
    def edge_ids(self) -> list[str]:
        return list(self._edges)

    @property
    def topics(self) -> dict[str, str]:
        return {
            edge_id: metadata.topic
            for edge_id, metadata in self._edges.items()
        }

    def __iter__(self) -> Iterator[str]:
        return iter(self._edges)

    def __getitem__(self, edge_id: str) -> EdgeMetadata:
        return self._edges[edge_id]

    @staticmethod
    def _make_camera(
        camera: dict[str, Any],
        profiles: dict[str, Any],
        translation_scale: float,
    ) -> dict[str, Any]:
        calibration = profiles[camera["calibration_profile"]]
        matrix = np.asarray(calibration["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(
            calibration["distortion_coefficients"],
            dtype=np.float64,
        )
        rotation_vector = np.asarray(
            calibration["rotation_vector"],
            dtype=np.float64,
        ).reshape(3, 1)
        translation_vector = np.asarray(
            calibration["translation_vector"],
            dtype=np.float64,
        ).reshape(3, 1)

        rotation, _ = cv2.Rodrigues(rotation_vector)
        position = -rotation.T @ translation_vector * translation_scale
        resolution = camera["resolution"]

        return {
            "id": camera["local_camera_id"],
            "name": camera["id"],
            "key": camera["key"],
            "resolution": (resolution["width"], resolution["height"]),
            "fps": camera["fps"],
            "R": rotation,
            "T": position,
            "fx": matrix[0, 0],
            "fy": matrix[1, 1],
            "cx": matrix[0, 2],
            "cy": matrix[1, 2],
            "k": distortion[[0, 1, 4]].reshape(3, 1),
            "p": distortion[[2, 3]].reshape(2, 1),
        }

    @staticmethod
    def _make_transform(cfg: Any, resolution: tuple[int, int]) -> np.ndarray:
        original_size = np.asarray(resolution)
        image_size = np.asarray(cfg.NETWORK.IMAGE_SIZE)
        center = original_size / 2
        scale = get_scale(original_size, image_size)
        return get_affine_transform(center, scale, 0, image_size)


if __name__ == "__main__":
    # 사용 예시:
    # 프로젝트 루트에서 아래 명령으로 실행한다.
    # python -m apps.server_worker.src.utils.edgemetadata
    from apps.server_worker.src.pose.core.config import config as sp3d_config

    metadata_loader = EdgeMetadataLoader(
        cfg=sp3d_config,
        edge_ids=["0_edge", "1_edge"],
    )

    print("edge_ids:", metadata_loader.edge_ids)
    print("topics:", metadata_loader.topics)

    for edge_id in metadata_loader:
        metadata = metadata_loader[edge_id]
        print(f"\n[{metadata.id}]")
        print("topic:", metadata.topic)
        print("transform shape:", metadata.transform.shape)

        for camera in metadata.cams:
            print(
                "camera:",
                camera["id"],
                camera["name"],
                camera["key"],
                camera["resolution"],
                f'{camera["fps"]}fps',
            )
