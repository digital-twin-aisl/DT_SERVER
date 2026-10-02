# SPDX-FileCopyrightText: 2025-2026 Electronics and Telecommunications Research Institute (ETRI) and Sejong University
# SPDX-License-Identifier: LGPL-2.1-or-later
import cv2
import numpy as np
import argparse


def calibrate_camera_from_video(
    video_source,
    checkerboard_size=(7, 10),
    square_size=30.0,
    frame_interval=10,
    min_samples=15
):
    """
    video_source:
        0, 1, ...      -> webcam
        "video.mp4"    -> video file

    checkerboard_size:
        체커보드 내부 코너 개수 (columns, rows)
        예: (9, 6)

    square_size:
        체커보드 한 칸 크기.
        단위는 자유지만 결과의 translation vector에도 동일 단위가 사용됨.
        예: 25.0 mm

    frame_interval:
        몇 프레임마다 체커보드를 검출할지 결정

    min_samples:
        캘리브레이션에 사용할 최소 이미지 개수
    """

    cols, rows = checkerboard_size

    # 체커보드의 3D 좌표
    # Z = 0 평면이라고 가정
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size

    # 실제 3D 좌표
    objpoints = []

    # 영상에서 검출된 2D 좌표
    imgpoints = []

    # 코너 보정 조건
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001
    )

    cap = cv2.VideoCapture(video_source)

    if not cap.isOpened():
        raise RuntimeError(f"영상 입력을 열 수 없습니다: {video_source}")

    frame_count = 0
    image_size = None

    print("카메라 캘리브레이션을 시작합니다.")
    print("ESC 또는 q: 종료")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("영상 입력이 종료되었습니다.")
            break

        frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        image_size = gray.shape[::-1]

        display = frame.copy()

        # 모든 프레임을 검사하지 않고 일정 간격으로 검사
        if frame_count % frame_interval == 0:

            found, corners = cv2.findChessboardCorners(
                gray,
                checkerboard_size,
                flags=(
                    cv2.CALIB_CB_ADAPTIVE_THRESH
                    + cv2.CALIB_CB_NORMALIZE_IMAGE
                )
            )

            if found:
                # sub-pixel 수준으로 corner 정확도 향상
                refined_corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria
                )

                objpoints.append(objp.copy())
                imgpoints.append(refined_corners)

                cv2.drawChessboardCorners(
                    display,
                    checkerboard_size,
                    refined_corners,
                    found
                )

                print(
                    f"Calibration sample: "
                    f"{len(objpoints)} / {min_samples}"
                )

        cv2.putText(
            display,
            f"Samples: {len(objpoints)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        cv2.imshow("Camera Calibration", display)

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(objpoints) < min_samples:
        print(
            f"\n샘플이 부족합니다. "
            f"{len(objpoints)}개 검출됨, 최소 {min_samples}개 권장."
        )

    if len(objpoints) < 3:
        raise RuntimeError("캘리브레이션에 필요한 체커보드 이미지가 너무 적습니다.")

    # 카메라 캘리브레이션
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None
    )

    print("\n========== Calibration Result ==========")

    print("\nRMS reprojection error:")
    print(rms)

    print("\nCamera Matrix K:")
    print(camera_matrix)

    print("\nDistortion Coefficients:")
    print(dist_coeffs)

    # 평균 reprojection error 계산
    total_error = 0

    for i in range(len(objpoints)):
        projected_points, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )

        error = cv2.norm(
            imgpoints[i],
            projected_points,
            cv2.NORM_L2
        ) / len(projected_points)

        total_error += error

    mean_error = total_error / len(objpoints)

    print("\nMean Reprojection Error:")
    print(mean_error)

    # 결과 저장
    np.savez(
        "camera_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms=rms,
        mean_reprojection_error=mean_error,
        image_size=image_size
    )

    print("\n결과 저장 완료:")
    print("camera_calibration.npz")

    return camera_matrix, dist_coeffs


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="영상 파일 경로. 입력하지 않으면 webcam 사용"
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=9,
        help="체커보드 가로 내부 코너 개수"
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=6,
        help="체커보드 세로 내부 코너 개수"
    )

    parser.add_argument(
        "--square",
        type=float,
        default=25.0,
        help="체커보드 한 칸 크기 (예: mm)"
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="검출할 프레임 간격"
    )

    args = parser.parse_args()

    if args.video is None:
        source = 0
    else:
        source = args.video

    calibrate_camera_from_video(
        source,
        checkerboard_size=(args.cols, args.rows),
        square_size=args.square,
        frame_interval=args.interval
    )