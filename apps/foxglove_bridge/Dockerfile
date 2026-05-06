FROM ros:humble-ros-base

RUN apt-get update && apt-get install -y \
    ros-humble-foxglove-bridge \
    && rm -rf /var/lib/apt/lists/*

CMD ["bash", "-c", "source /opt/ros/humble/setup.bash && ros2 launch foxglove_bridge foxglove_bridge_launch.xml port:=8765"]
