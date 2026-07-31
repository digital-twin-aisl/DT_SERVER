#/bin/bash
apt update
pip install -r requirements_root.txt

# install torch2TRT
git clone https://github.com/NVIDIA-AI-IOT/torch2trt.git
cd torch2trt
python3 setup.py install
cd ..
rm -rf torch2trt

python3 src/utils/download_from_drive.py