# ArUco Marker Tree Generator

Isaac Sim UI에서 OpenCV ArUco 딕셔너리, 마커 ID, 마커 한 변 길이(mm)를 입력해
여러 정사각형 마커 보드를 하나의 재사용 가능한 USD 트리에 추가하는
Extension입니다.

## Marker tree 구조

기본 관리 대상은 개별 보드가 아니라 다음 트리 USD입니다.

```text
apps/isaac_sim_client/aruco_boards/aruco_marker_tree.usd
```

`Add marker to tree`를 누를 때마다 기존 트리에 마커가 자식으로 추가됩니다.
개별 보드의 geometry, material, metadata는 모두 tree USD 내부의 marker Prim 아래에
저장됩니다. 따라서 별도 marker USD는 생성되지 않으며, 삭제·수정은 tree 안의 해당
marker Prim 하나를 관리하면 됩니다. ArUco 이미지는 USD의 texture asset 특성상 트리
옆 `textures/`에 PNG로 저장됩니다. 트리 경로를 다르게 지정하면 용도별로 여러 마커
컬렉션을 만들 수 있습니다.

하나의 트리 안에서 `(dictionary, marker ID)` 조합은 영상에서 검출 가능한 GCP의
고유 식별자입니다. 같은 조합을 다시 추가하면 모호한 대응을 방지하기 위해 오류로
처리됩니다.

```text
/ArUcoMarkerTree
└── Markers
    ├── Marker_4X4_50_0
    ├── Marker_4X4_50_1
    └── Marker_6X6_250_12
```

`Camera placement > In front`가 켜져 있으면 새 마커 하나의 카메라 앞 위치와
회전이 해당 마커 자식 transform으로 tree USD 안에 저장됩니다. 옵션을 끄면
마커는 추가 순서대로 10 mm 간격을 두고 좌우로 자동 배치됩니다.

`Preview tree in current Stage session`이 켜져 있으면 현재 campus Stage의 익명
session layer에만 tree reference를 만듭니다. campus USD root layer에는 Prim,
reference, marker metadata 또는 transform override를 기록하지 않으며 Extension은
campus USD를 저장하지 않습니다. 미리보기 경로는
`/World/ArUcoMarkerTreeEditor/<tree 이름>/Tree`입니다.
campus에 동일 tree를 가리키는 고정 reference가 이미 있으면 중복 미리보기를 만들지
않고 기존 reference를 사용하며, gizmo 수정값만 session layer에 기록합니다.

새 마커는 자동으로 선택됩니다. Move/Rotate gizmo로 배치한 뒤 `Save selected pose`를
누르면 composed marker transform을 tree USD의 해당 marker Prim에 직접 저장하고
session override를 제거합니다. `Apply fields + pose`는 현재 Dictionary, Marker ID,
Marker length 입력값과 gizmo transform을 함께 tree USD에 반영합니다. Dictionary와
ID를 바꾸면 marker Prim 이름과 texture도 새 식별자에 맞춰 갱신됩니다.

`Delete selected`는 선택한 marker Prim을 tree USD에서 제거하고 해당 marker가
사용하던 관리 대상 PNG도 삭제합니다. 따라서 추가, 배치, 속성 수정, 삭제의 영구
저장 대상은 항상 tree USD 하나이며 campus USD는 고정 상태로 유지됩니다.

기본으로 켜진 `Camera placement > In front`는 마커를 추가할 때 활성
Viewport 카메라의 1 m 앞에 정면이 카메라를 향하도록 배치합니다. `Distance (m)`로
거리를 바꿀 수 있습니다. 계산된 위치와 회전은 새 marker의 tree transform으로
즉시 저장됩니다. 이후 gizmo로 조정한 값도 `Save selected pose`를 통해 같은 tree
transform을 갱신합니다.

보드는 Z-up, meters-per-unit 1.0으로 작성됩니다. 중심은 로컬 원점이며 윗면은
`+Z`, 기본 두께는 1 mm입니다. 흰색 backing에는 static collision이 적용되고,
입력한 길이는 검은색 ArUco 마커 외곽의 실제 한 변 길이입니다. 흰색 여백을
포함한 전체 보드의 한 변 길이는 `입력한 마커 길이 × 1.05`로 생성됩니다. 따라서
마커 길이가 100 mm이면 전체 보드는 105 mm이며, 각 방향 여백은 2.5 mm입니다.

## 설치 및 실행

Isaac Sim의 Extension Manager에서 아래 디렉터리를 Extension Search Path에
추가합니다.

```text
<DT_SERVER>/apps/isaac_sim_client/exts
```

그 다음 `ArUco Marker Tree Generator`를 검색하여 활성화합니다. 창이 뜨지 않으면
Extension을 한 번
비활성화 후 다시 활성화하면 됩니다. 현재 버전은 활성화 시 창을 바로 엽니다.

Isaac Sim Python의 OpenCV에 `cv2.aruco`가 없다면 Isaac Sim 설치 디렉터리에서
다음을 실행한 뒤 재시작합니다.

```bash
./python.sh -m pip install opencv-contrib-python-headless
```

기존 OpenCV 패키지가 있는 환경에서는 패키지 충돌 가능성이 있으므로 먼저
아래 명령으로 `aruco` 포함 여부를 확인하는 것을 권장합니다.

```bash
./python.sh -c "import cv2; print(cv2.__version__, hasattr(cv2, 'aruco'))"
```

## 스크립트에서 호출

Extension이 활성화된 Isaac Sim Python 환경에서는 UI 없이도 호출할 수 있습니다.

```python
from meta_sejong.aruco_board.generator import (
    append_marker_to_tree_usd,
    read_marker_tree_usd,
)

append_marker_to_tree_usd(
    dictionary_name="DICT_6X6_250",
    marker_id=23,
    marker_length_mm=120.0,
    tree_path="/tmp/aruco_marker_tree.usd",
)

for marker in read_marker_tree_usd("/tmp/aruco_marker_tree.usd"):
    print(
        marker.dictionary_name,
        marker.marker_id,
        marker.marker_length_mm,
        marker.local_transform,
    )
```
