FROM ros:humble-ros-base

WORKDIR /app

# ROS 2 환경 위에 Python 패키지 설치를 위한 pip 세팅
RUN apt-get update && apt-get install -y python3-pip && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./app ./app

# 컨테이너 실행 시 ROS 2 환경변수를 source하고, rclpy 노드(ros2_main.py)를 실행합니다.
CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && python3 -m app.ros2_main"]
