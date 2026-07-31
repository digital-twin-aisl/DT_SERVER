# Edge Digital Twin

## Requirements
```bash
python : 3.10.12
torch : 2.1.0
cuda : 12.2 
tensorRT : 8.6.2
opencv-python : 4.x
numpy : 1.x
```

## TensorRT Anaconda 연결
```bash
# TensorRT 환경 설정 명령어 추가 예정
```

## unified_env 실행 전
```bash
$ export PYTHONPATH=/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
```

## Camera Calibration

### Intrinsic Parameter Calibration (intrinsic.py)
이 모듈은 체커보드 패턴을 사용하여 카메라의 내적 매개변수를 캘리브레이션합니다.

#### 사용 방법
```bash
# 카메라 캘리브레이션 실행
python tools/intrinsic.py

# 실행 중 조작법:
# - 'c' 키: 현재 체커보드 데이터 추가 (20개 이상 권장)
# - 'q' 키: 프로그램 종료 및 캘리브레이션 분석
```

#### 출력 결과
- **카메라 매트릭스**: 초점거리(fx, fy), 주점(cx, cy)
- **왜곡 계수**: 방사 및 접선 왜곡 매개변수
- **RMS 재투영 오차**: 캘리브레이션 품질 지표
- **안정성 분석**: 변동계수로 신뢰성 평가

#### 지원 카메라
- USB 카메라 (인덱스 번호)
- IP 카메라 (스트리밍 URL)

#### 신뢰성 기준
- **매우 안정적**: 변동계수 < 1.0%
- **보통 수준**: 변동계수 < 2.0%
- **불안정**: 변동계수 ≥ 2.0% (데이터 재수집 필요)

#### 체커보드 패턴 출처
- [체커보드 패턴 이미지](https://github.com/opencv/opencv/blob/4.x/doc/pattern.png)

### Extrinsic Parameter Calibration (extrinsic.py)
이 모듈은 ArUco 마커를 사용하여 카메라의 외적 매개변수(위치 및 자세)를 캘리브레이션합니다.

#### 사용 방법
```bash
# 외부 파라미터 캘리브레이션 실행
python tools/extrinsic.py

# 실행 중 조작법:
# - 's' 키: 현재 검출된 ArUco 마커의 외부 파라미터 저장
# - 'q' 키: 프로그램 종료
```

#### 필수 조건
- 사전에 `intrinsic.py`로 내부 파라미터 캘리브레이션 완료 필요
- ArUco 마커 (DICT_6X6_250 딕셔너리 사용)
- 마커의 정확한 실제 크기(미터 단위) 정보

#### 출력 결과
- **회전 벡터(rvec)**: 카메라에서 마커까지의 회전
- **변환 벡터(tvec)**: 카메라에서 마커까지의 위치 (x, y, z)
- **마커 ID**: 검출된 ArUco 마커 식별번호
- **측정 거리**: 카메라와 마커 간의 유클리드 거리

#### 지원 카메라
- USB 카메라 (인덱스 번호)
- IP 카메라 (스트리밍 URL)

#### ArUco 마커 생성
ArUco 마커는 온라인 생성기를 통해 생성할 수 있습니다:
- [ArUco Marker Generator](https://chev.me/arucogen/)
- 딕셔너리: 7x7 (250 markers)

#### 사용 팁
- 마커를 평평한 표면에 부착하여 사용
- 조명이 균일한 환경에서 측정
- 마커가 화면에 명확하게 보이는 거리에서 측정

## Structure diagram

![Structure](./data/image.png)

## 파일 구조
```
Edge_DT/
├── tools/
│   └── extrinsic.py               # 카메라 외적 매개변수 캘리브레이션
│   └── intrinsic.py               # 카메라 내적 매개변수 캘리브레이션
├── camera_calibration_results.json # 캘리브레이션 결과 저장
├── data/
│   ├── pattern.png                 # 체커보드 패턴 이미지
│   └── image.png                   # 구조도
└── README.md
```