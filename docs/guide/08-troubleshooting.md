# 8. 문제 해결

먼저 다음 세 가지를 확인합니다.

```bash
python -m apps.edge_manager region status <region> --json   # 어떤 증거가 빠졌는지
python -m apps.edge_manager region logs <region>             # 서버 로그
python -m apps.edge_manager region logs <region> --edge-id edge_1
```

## 설치

| 증상 | 원인과 조치 |
| --- | --- |
| `torch.cuda.is_available()`가 `False` | 드라이버와 PyTorch CUDA wheel 조합이 맞지 않음. PyTorch 선택기로 다시 설치 |
| `cv2`에 `aruco`가 없음 | 다른 OpenCV 패키지가 섞임. `pip uninstall opencv-python opencv-contrib-python opencv-python-headless` 후 `requirements.txt` 재설치 |
| `numpy` 2.x 관련 오류 | VGGT-Omega와 OpenCV 4.11 이하는 `numpy<2` 필요. `requirements.txt`의 범위를 지킴 |
| `hashlib.file_digest`, `enterContext` AttributeError | Python 3.10에서 오래된 코드 사용. 최신 버전으로 업데이트 |
| 뷰어가 빈 화면, `/static/dist/viewer.js` 404 | `npm run build --prefix apps/frontend_api/web`를 실행하지 않음 |
| 수동 보정 편집기 테스트에서 `three`를 찾을 수 없음 | `apps/frontend_api/web`에서 `npm ci` 후 `build_manual_editor.sh` 실행 |

## 연결

| 증상 | 원인과 조치 |
| --- | --- |
| `Scene subscriber reconnecting: Unable to connect` | router(`zenohd`)가 없거나 주소가 다름. `SCENE_ZENOH_ENDPOINT`, 방화벽 TCP 7447 확인 |
| 엣지가 `list`에 보이지 않음 | agent의 `--endpoint`가 서버 router를 가리키는지, `agent.py run`이 실행 중인지 확인 |
| manager 시작 시 topic root 오류 | manager `--topic-root`와 모든 manifest의 `topic_root`가 같아야 함 |
| manager가 외부 바인딩을 거부 | `DT_MANAGER_TOKEN`을 설정하거나 loopback + SSH 터널 사용 |
| CLI `401` | `DT_MANAGER_TOKEN`을 CLI 셸에도 설정 |

## 구역 실행

| 증상 | 원인과 조치 |
| --- | --- |
| 계속 `starting` | 프로세스는 떴지만 유효 관측이 없음. 엣지 카메라 상태, `max_input_age`, 시계 동기화(NTP) 확인 |
| `degraded` | 일부 엣지 입력 누락. 해당 엣지 로그와 카메라 RTSP 연결 확인 |
| `failed` (설정 검증) | manifest·보정·ground 경로, 서버 프로파일의 deployment 일치 확인 |
| 서버 로그 `expected ZNH2` | 엣지와 서버 버전 불일치. 둘 다 같은 버전으로 업그레이드 |
| 서버가 패킷을 calibration digest/workspace 불일치로 거부 | 호스트마다 manifest·보정·ground 캐시가 다름. 같은 커밋으로 맞춤 |
| 엣지가 `ultralytics` 오류로 종료 | ReID 요청 + Ultralytics 미설치. `--no-reid`로 실행하거나 라이선스 검토 후 설치 |
| 엣지가 30초 후 스스로 추론 중지 | manager와 연결 끊김(lease 만료). 네트워크와 manager 상태 확인 |
| GPU OOM | 보정 worker와 추론을 동시에 실행하지 않음. `lod2_count`를 줄임 |
| `POC_posenet.pth.tar` 없음 | 가중치는 저장소에 없음. `DT_POSENET_URL`로 받거나 직접 복사 |

## 결과 품질

| 증상 | 확인할 것 |
| --- | --- |
| 사람 위치가 벽 안이나 공중에 있음 | 보정(extrinsic)과 지면 캐시. 수동 보정 편집기로 지도와 영상 정렬 확인 |
| ID가 자주 바뀜 | ReID 꺼짐(위치 기반 ID) 또는 카메라 겹침 부족. AOI와 `min_views` 확인 |
| 자세가 일부 사람만 나옴 | 정상. `lod2_count`만큼만 LOD 2를 계산함. `lod_policy=all`로 모두 계산 가능(GPU 부하 증가) |

해결되지 않으면 [SUPPORT.md](../../SUPPORT.md)의 양식으로 이슈를 남겨 주세요.
