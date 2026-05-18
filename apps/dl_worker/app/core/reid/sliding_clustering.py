# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import os
import json
import shutil
import time
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from itertools import product, cycle
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


class ClusteringSliding:
    """
    Sliding-window 기반 실시간(또는 준-실시간) ReID 클러스터링 파이프라인.

    동작 개요
    1) 외부에서 frame 단위 JSON을 지속적으로 `process_json()`으로 넣음.
       - JSON 스키마(리스트 형태):
         {
           "zone": <int>,     # 존 번호
           "cam": <int>,      # 카메라 번호
           "frame": <int>,    # 프레임 번호
           "bbox": [x, y, w, h],
           "feature": [f1, f2, ..., f128]  # 128차원 appearance feature
         }
    2) 내부 deque 버퍼(frame_buffer)에 최근 window_size개의 프레임을 유지.
    3) 버퍼가 가득 차면 `cluster_and_save()`가 실행되어:
       - DBSCAN으로 현재 윈도우의 특징들을 군집화
       - 이전 윈도우의 클러스터 중심과 코사인 유사도로 global ID 매칭
       - 결과를 결과 디렉토리에 frame_<index>.json으로 저장

    사용 시 주의
    : eps/min_samples 및 파라미터 변경 금. 지.
    """

    def __init__(self, base_dir, zones, window_size=10, data=None): #파이프라인 초기화
        self.base_dir = base_dir
        self.data_dir = os.path.join(base_dir, "data")
        self.result_dir = os.path.join(base_dir, "results")

        # os.makedirs(self.result_dir, exist_ok=True)
        self.data = data
        self.zones = zones
        self.window_size = window_size   
        self.frame_counter = 0 # 윈도우 인덱싱용 (처리된 프레임 수)

        #최근 window_size개의 프레임을 유지하는 버퍼
        self.frame_buffer = deque(maxlen=window_size) #튜플

        #centroid 기반 ID 매칭용 상태 변수
        self.previous_centroids = {}
        self.next_global_id = 0

        # 시각화 스타일 (시각화용 코드)
        """
        self.id_style_map = {}
        self._colors = list(plt.cm.tab20.colors)[:20]
        self._markers = ["o", "s", "^", "*", "D", "v", "<", ">", "X", "P",
                         "H", "d", "1", "2", "3", "4", "|", "_", "+", "."]
        assert len(self._colors) == len(self._markers)
        self._max_unique = len(self._colors)
        self._next_unique_index = 0
        self._fallback_cycle = cycle(list(product(self._colors, self._markers)))

    def _get_style_for_id(self, gid: int):
        if gid not in self.id_style_map:
            if self._next_unique_index < self._max_unique:
                idx = self._next_unique_index
                self.id_style_map[gid] = (self._colors[idx], self._markers[idx])
                self._next_unique_index += 1
            else:
                self.id_style_map[gid] = next(self._fallback_cycle)
        return self.id_style_map[gid]
        """

    def process_json(self, json_path):
        """
        단일 프레임 JSON을 읽어 내부 버퍼에 추가.
        버퍼가 가득 차면 클러스터링 수행.

        - JSON이 비어있어도 버퍼에 빈 리스트로 추가됨.
        - frame_counter 증가 카운트, window_size 도달 시 클러스터링.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        frame_feats, frame_meta = [], []
        if data:
            for item in data:
                feat = np.array(item["feature"])
                frame_feats.append(feat)
                frame_meta.append({
                    "zone": int(item["zone"]),
                    "cam": int(item["cam"]),
                    "frame": int(item["frame"]),
                    "bbox": item["bbox"],
                    "feature": item["feature"]
                })
        self.frame_buffer.append((frame_feats, frame_meta))

        self.frame_counter += 1

        if self.frame_counter >= self.window_size:
            self.cluster_and_save()
    def process_realtime(self, data):

        frame_feats, frame_meta = [], []
        if data:
            for item in data:
                # feature가 None이 아닌지 확인
                if item.get("feature") is not None:
                    feat = np.array(item["feature"])
                    frame_feats.append(feat)
                    frame_meta.append({
                        "zone": int(item["zone"]),
                        "cam": int(item["cam"]),
                        "frame": int(item["frame"]),
                        "bbox": item["bbox"],
                        "feature": item["feature"]
                    })
        
        # frame_buffer에 데이터 추가
        self.frame_buffer.append((frame_feats, frame_meta))
        self.frame_counter += 1
        # 윈도우가 가득 차면 클러스터링 수행
        if self.frame_counter >= self.window_size:
            results = self.cluster()
            if results:
                return results
                ids = [item["global_id"] for item in results]
                bbox = [item["bbox"] for item in results]
                return ids, bbox
        
        # 윈도우가 아직 가득 차지 않았거나 결과가 없는 경우 빈 리스트 반환
        return None
            
    def cluster(self):
        features, meta_all = [], []
        for feats, metas in self.frame_buffer:
            if feats: 
                features.extend(feats)
                meta_all.extend(metas)
        # print(self.frame_buffer)
        if len(features) == 0:
            return

        features = np.array(features)
    
        dbscan = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
        labels = dbscan.fit_predict(features)

        cluster_to_indices = {}
        for idx, label in enumerate(labels):
            if label == -1 or np.isnan(label):
                continue
            cluster_to_indices.setdefault(label, []).append(idx)

        current_centroids = {
            label: np.mean(features[indices], axis=0)
            for label, indices in cluster_to_indices.items()
        }

        label_to_global_id = {}
        used_global_ids = set()
        for curr_label, curr_centroid in current_centroids.items():
            best_match, best_score = None, -1
            for prev_id, prev_centroid in self.previous_centroids.items():
                if prev_id in used_global_ids:
                    continue
                sim = cosine_similarity(
                    curr_centroid.reshape(1, -1),
                    prev_centroid.reshape(1, -1)
                )[0][0]
                if sim > best_score and sim > 0.8:
                    best_score = sim
                    best_match = prev_id
            if best_match is not None: #기존 id 할당
                label_to_global_id[curr_label] = best_match
                used_global_ids.add(best_match)
            else: # 새로운 아이디 할당
                label_to_global_id[curr_label] = self.next_global_id
                self.next_global_id += 1

        self.previous_centroids = { # 현재 시점 centroid 를 키로 저장
            label_to_global_id[label]: centroid 
            for label, centroid in current_centroids.items()
        }

        results = []
        for idx, meta in enumerate(meta_all):
            label = labels[idx]
            if label == -1 or label not in label_to_global_id:
                continue
            global_id = label_to_global_id[label]
            results.append({
                "global_id": global_id,
                **meta
            })
    

        return results

    def cluster_and_save(self):
        """
        1. 현재 슬라이딩 윈도우의 모든 feature를 DBSCAN으로 군집화.
        2. 이전 윈도우와의 코사인 유사도 기반 매칭으로 global_id 부여,
        3. 결과를 {result_dir}/frame_<end_frame>.json에 저장.

        Pipeline
        1) frame_buffer 내 모든 feature/metadata를 펼침.
        2) fixed param 기준으ㅗ DBSCAN으로 클러스터링.
           - label == -1은 노이즈로 취급하고 결과에서 제외.
        3) 각 클러스터의 centroid 벡터 계산.
        4) 이전 윈도우의 global_id 중심과 코사인 유사도 매칭
           - 매칭되면 기존 global_id 재사용
           - 없으면 새로운 global_id 할당
        5) 매칭된 global_id로 결과 리스트를 생성 및 저장. 

        Side Effects
        - result_dir에 frame_<index>.json 파일을 생성.
        - previous_centroids 상태를 현재 윈도우 기준으로 업데이트.
        """
        start_time = time.time()

        features, meta_all = [], []
        for feats, metas in self.frame_buffer:
            if feats: 
                features.extend(feats)
                meta_all.extend(metas)

        if len(features) == 0:
            return

        features = np.array(features)
        dbscan = DBSCAN(eps=0.05, min_samples=5, metric="cosine")
        labels = dbscan.fit_predict(features)

        cluster_to_indices = {}
        for idx, label in enumerate(labels):
            if label == -1 or np.isnan(label):
                continue
            cluster_to_indices.setdefault(label, []).append(idx)

        current_centroids = {
            label: np.mean(features[indices], axis=0)
            for label, indices in cluster_to_indices.items()
        }

        label_to_global_id = {}
        used_global_ids = set()
        for curr_label, curr_centroid in current_centroids.items():
            best_match, best_score = None, -1
            for prev_id, prev_centroid in self.previous_centroids.items():
                if prev_id in used_global_ids:
                    continue
                sim = cosine_similarity(
                    curr_centroid.reshape(1, -1),
                    prev_centroid.reshape(1, -1)
                )[0][0]
                if sim > best_score and sim > 0.8:
                    best_score = sim
                    best_match = prev_id
            if best_match is not None: #기존 id 할당
                label_to_global_id[curr_label] = best_match
                used_global_ids.add(best_match)
            else: # 새로운 아이디 할당
                label_to_global_id[curr_label] = self.next_global_id
                self.next_global_id += 1

        self.previous_centroids = { # 현재 시점 centroid 를 키로 저장
            label_to_global_id[label]: centroid 
            for label, centroid in current_centroids.items()
        }

        results = []
        for idx, meta in enumerate(meta_all):
            label = labels[idx]
            if label == -1 or label not in label_to_global_id:
                continue
            global_id = label_to_global_id[label]
            results.append({
                "global_id": global_id,
                **meta
            })

        end_frame = self.frame_counter - 1  
        json_path = os.path.join(self.result_dir, f"frame_{end_frame:05d}.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        return results
        # elapsed_time = time.time() - start_time
        # print(f"{json_path} | ID 수: {len(set(label_to_global_id.values()))} | 처리 시간: {elapsed_time:.3f}초")

    def collect_all_frame_numbers(self): # 인풋 관련 코드 (인풋 형식에 맞게 수정 필요)
        # data 디렉토리의 파일명들 스캔, 사용 가능한 프레임 번호 수집, 모든 존에 대해 순회 진행
        frame_nums = set()
        for zone in self.zones:
            original_path = os.path.join(self.data_dir, f"{zone}_original")
            files = [f for f in os.listdir(original_path) if f.endswith(".json")]
            for fname in files:
                try:
                    num = int(fname.replace("frame_", "").replace(".json", ""))
                    frame_nums.add(num)
                except:
                    continue
        return sorted(frame_nums)

    def get_next_frame_paths(self, index):
        next_paths = []
        for zone in self.zones:
            original_path = os.path.join(self.data_dir, f"{zone}_original")
            fname = f"frame_{index:05d}.json"
            fpath = os.path.join(original_path, fname)
            if os.path.exists(fpath):
                active_path = os.path.join(self.data_dir, f"{zone}_reid_json", fname)
                os.makedirs(os.path.dirname(active_path), exist_ok=True)
                shutil.copy(fpath, active_path)
                next_paths.append((zone, index, active_path))
        return next_paths

    def finalize(self):
        print("모든 프레임 데이터 클러스터링 완료.")
