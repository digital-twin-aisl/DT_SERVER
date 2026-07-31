import gdown
import os
import os.path as osp
import zipfile

PWD = osp.dirname(osp.abspath(__file__))

# Google Drive 파일 ID
data0417_zip_id = '14f5_bPqjiLOS09HykQ9XM4-99pWqu-mx'
data0417_2_zip_id = '10i8ewHc9wc2COJiAxYKVB_9bL61kis62'

# 경로 설정
project_root = osp.abspath(osp.join(PWD, '..', '..'))
target_data_dir = osp.join(project_root, 'data')
#output_zip_path_1 = osp.join(target_data_dir, 'data_0417.zip')
output_zip_path_2 = osp.join(target_data_dir, 'data_0417_2.zip')
extract_dir = target_data_dir

def download_from_google_drive(file_id, output_path):
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output_path, quiet=False)

def unzip_file(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"압축 해제 완료: {extract_to}")

def run():
    '''
    # data_0417
    if not osp.exists(osp.join(extract_dir, 'data_0417')):
        print("data_0417.zip 다운로드 중...")
        download_from_google_drive(data0417_zip_id, output_zip_path_1)

        print("압축 해제 중...")
        unzip_file(output_zip_path_1, extract_dir)

        print("ZIP 파일 삭제 중...")
        os.remove(output_zip_path_1)
    else:
        print("data_0417 폴더가 이미 존재합니다.")
    #'''

    #'''    
    # data_0417_2
    if not osp.exists(osp.join(extract_dir, 'data_0417_2')):
        print("data_0417_2.zip 다운로드 중...")
        download_from_google_drive(data0417_2_zip_id, output_zip_path_2)

        print("압축 해제 중...")
        unzip_file(output_zip_path_2, extract_dir)

        print("ZIP 파일 삭제 중...")
        os.remove(output_zip_path_2)
    else:
        print("data_0417_2 폴더가 이미 존재합니다.")
    #'''

    print("완료: 모든 데이터가 data/에 저장되었습니다.")

if __name__ == '__main__':
    run()
