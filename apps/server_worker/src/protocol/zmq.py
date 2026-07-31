import zmq
import pickle
import numpy as np
from typing import Any, Optional, Dict, List


class Protocol:
    def __init__(self, host: str, port: int):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUSH)
        self.socket.connect(f"tcp://{host}:{port}")
        print(f"서버 tcp://{host}:{port} 에 연결되었습니다.")

    def send_scene(self, scene: Dict[str, Any]) -> bool:
        """버전이 명시된 JSON scene payload를 downstream 프로세스에 전송한다."""
        try:
            self.socket.send_json(scene)
            return True
        except Exception as exc:
            print("Scene 전송 실패:", exc)
            return False

    def send_result(
        self,
        data: Dict[str, Any],
        ids: List[List[Optional[int]]],
        timestamps: List[float],
        edge_ids: List[str],
        lod: Optional[Dict[str, int]] = None,
    ) -> bool:
        """
        모든 Edge 결과를 한 번에 전송 (pickle 사용)
        lod[edge_id]==2: pred에서 0만 있는 항목 제거 후 전송
        lod[edge_id]==1: grid_centers 전송
        else: None
        """
        try:
            pred = data.get("pred")[:, :, :, :3].cpu().numpy()
            raw_grid_centers = data.get("grid_centers").cpu().numpy()
            grid_centers = raw_grid_centers[:, :, :3]
            valid_roots = raw_grid_centers[:, :, 3] >= 0
            mesh = data.get("mesh", None)
            results: List[Dict[str, Any]] = []
            lod = lod or {}
            idx = 0

            for index, edge_id in enumerate(edge_ids):
                if edge_id is None:
                    continue

                result = None
                gid = None
                edge_lod = lod.get(edge_id, 2)
                if edge_lod == 3:
                    if pred is not None and idx < len(pred) and pred[idx] is not None:
                        mask = valid_roots[idx] & (
                            np.abs(pred[idx]).sum(axis=(1, 2)) != 0
                        )
                        edge_mesh = (
                            mesh[idx] if mesh is not None and idx < len(mesh) else None
                        )
                        result = {"pred": pred[idx][mask], "mesh": edge_mesh}
                        if ids is not None and idx < len(ids) and ids[idx] is not None:
                            gid = [ids[idx][i] for i, keep in enumerate(mask) if keep]
                elif edge_lod == 2:
                    if pred is not None and idx < len(pred) and pred[idx] is not None:
                        mask = valid_roots[idx] & (
                            np.abs(pred[idx]).sum(axis=(1, 2)) != 0
                        )
                        result = pred[idx][mask]
                        if ids is not None and idx < len(ids) and ids[idx] is not None:
                            gid = [ids[idx][i] for i, keep in enumerate(mask) if keep]
                    else:
                        result = None
                elif edge_lod == 1:
                    if grid_centers is not None and idx < len(grid_centers):
                        result = grid_centers[idx]
                    gid = ids[index] if ids and index < len(ids) else None
                else:
                    result = None
                    gid = ids[index] if ids and index < len(ids) else None

                entry = {
                    "edge_id": edge_id,
                    "time": timestamps[index]
                    if timestamps and index < len(timestamps)
                    else None,
                    "data": result,
                    "global_id": gid,
                }
                results.append(entry)
                idx += 1

            payload = {"results": results}
            # print(payload)
            self.socket.send(pickle.dumps(payload))
            return True

        except Exception as e:
            print("전송 실패:", e)
            return False

    def close(self):
        self.socket.close()
        self.context.term()
