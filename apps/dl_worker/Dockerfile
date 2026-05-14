FROM ros:humble-ros-base

WORKDIR /app

# ROS 2 환경 위에 Python 패키지 설치를 위한 pip 세팅
RUN apt-get update && apt-get install -y \
    python3-pip \
    ros-humble-tf2-ros \
    ros-humble-cv-bridge \
    ros-humble-rosidl-default-generators \
    && rm -rf /var/lib/apt/lists/*

COPY ./apps/dl_worker/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# 커스텀 인터페이스 패키지 복사 및 빌드
COPY ./interfaces /app/interfaces
RUN pip3 install "setuptools<=58.2.0" && . /opt/ros/humble/setup.sh && colcon build --packages-select dt_interfaces

COPY ./apps/dl_worker/app ./app
COPY ./apps/dl_worker/launch ./launch

# Phase 2 pose graph scripts (GTSAM wrapper)
COPY ./scripts/phase2_pose_graph /app/scripts/phase2_pose_graph

# Phase 3 online calibration scripts (SuperPoint+LightGlue overlap detection)
COPY ./scripts/phase3_online_calibration /app/scripts/phase3_online_calibration

# 컨테이너 실행 시 ROS 2 환경변수와 커스텀 메시지 환경변수를 source하고 실행합니다.
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && source /app/install/setup.bash && python3 -m app.ros2_main"]
