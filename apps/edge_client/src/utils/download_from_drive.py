import gdown
import os
import os.path as osp

# Download the example dataset and model file from Google Drive

PWD = osp.dirname(osp.abspath(__file__))

# example dataset google drive file id
video_ids = {
    'hd_00_01.mp4': '1KkdoIXyxKFbsh8NYHyaGgvtz3r6qsr6I',
    'hd_00_02.mp4': '1S1QibbEYD0JAMzwt5Zl1IwUuNYNb8z2O',
    'hd_00_03.mp4': '1s7jPNvglf_iH1rabsSve-POiK57mbz7-',
    'hd_00_04.mp4': '1x1q-2uGuGV1xRWvA-AZ19g8ZXsPcsBHx'
}

calibration_ids = {
    'camera1.pkl': '1STmgz9z6923DwFCuJ4QNaKgtMTRe7QcO',
    'camera2.pkl': '1eoeyJgZwOvTBMWwIoGb_QHW2xALFvKpF',
    'camera3.pkl': '12--O4zXge5qMNZCfCBZ9wTnuMN-i_Mrk',
    'camera4.pkl': '1mTmLY60Wc0eunuUAEKHJIyKNfW427Yvd'
}

model_ids = {
    'POC_posenet.pth.tar': '1q3D8VWAvR6fBcQEHRsFKL3nx1EJYwCaQ',
}

model_urls = {
    'market_bot_R50-ibn.pth': (
        'https://github.com/JDAI-CV/fast-reid/releases/download/'
        'v0.1.1/market_bot_R50-ibn.pth'
    ),
}

def download_from_google_drive(file_id, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def run():
    for video_name, video_id in video_ids.items():
        output_path = osp.join(PWD, '..', '..', 'data', 'data_0705', 'hdVideos', video_name)
        if osp.exists(output_path):
            print(f"{video_name} already exists")
            continue
        download_from_google_drive(video_id, output_path)
        print(f"Downloaded {video_name}")

    for calibration_name, calibration_id in calibration_ids.items():
        output_path = osp.join(PWD, '..', '..', 'data', 'data_0705', 'calibration', calibration_name)
        if osp.exists(output_path):
            print(f"{calibration_name} already exists")
            continue
        download_from_google_drive(calibration_id, output_path)
        print(f"Downloaded {calibration_name}")

    for model_name, model_id in model_ids.items():
        output_path = osp.join(PWD, '..', '..', 'models', model_name)
        if osp.exists(output_path):
            print(f"{model_name} already exists")
            continue
        download_from_google_drive(model_id, output_path)
        print(f"Downloaded {model_name}")

    for model_name, url in model_urls.items():
        output_path = osp.join(
            PWD,
            '..',
            'reid',
            'fast-reid',
            'weights',
            model_name,
        )
        if osp.exists(output_path):
            print(f"{model_name} already exists")
            continue
        os.makedirs(osp.dirname(output_path), exist_ok=True)
        gdown.download(url, output_path, quiet=False)
        print(f"Downloaded {model_name}")

    print("Download completed")

if __name__ == '__main__':
    run()
