#!/usr/bin/env python3
"""Register one camera in the edge-local YAML configuration."""

import argparse
from collections import deque
from getpass import getpass
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import threading
import time
from urllib.parse import quote, urlsplit, urlunsplit

import yaml


DEFAULT_CONFIG = Path(__file__).parent / "config" / "cameras.local.yaml"
DEFAULT_CAPTURE_DIR = Path(__file__).parent / "data" / "captures"
DEFAULT_RECORD_DIR = Path(__file__).parent / "data" / "recordings"
DEFAULT_EDGE_ID_FILE = Path(__file__).parent / "config" / "edge.local.json"


def build_rtsp_url(raw_url, username="", password=""):
    parsed = urlsplit(raw_url.strip())
    if parsed.scheme not in {"rtsp", "rtsps"} or not parsed.hostname:
        raise ValueError("rtsp:// 또는 rtsps:// 형식의 주소를 입력해주세요.")
    if parsed.username or parsed.password:
        raise ValueError("RTSP 주소에는 계정정보를 포함하지 마세요.")

    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    auth = ""
    if username:
        auth = f"{quote(username, safe='')}:{quote(password, safe='')}@"
    return urlunsplit(
        (parsed.scheme, f"{auth}{host}{port}", parsed.path or "/", parsed.query, "")
    )


def probe_rtsp(url, timeout_ms=8000):
    import cv2

    params = []
    if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, timeout_ms]
    if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
        params += [cv2.CAP_PROP_READ_TIMEOUT_MSEC, timeout_ms]

    capture = cv2.VideoCapture()
    try:
        original_stderr = os.dup(2)
        null_stderr = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(null_stderr, 2)
            opened = capture.open(url, cv2.CAP_FFMPEG, params)
            ok, frame = capture.read() if opened else (False, None)
        finally:
            os.dup2(original_stderr, 2)
            os.close(original_stderr)
            os.close(null_stderr)

        if not opened or not ok:
            raise RuntimeError("RTSP 첫 프레임을 읽을 수 없습니다.")

        fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
        fps = capture.get(cv2.CAP_PROP_FPS)
        return {
            "codec": codec.strip("\x00 ") or "확인 불가",
            "width": frame.shape[1],
            "height": frame.shape[0],
            "fps": round(fps, 2) if fps > 0 else "확인 불가",
        }
    finally:
        capture.release()


def load_config(config_path):
    if not config_path.exists():
        return {"CAMERAS": []}
    return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {"CAMERAS": []}


def load_cameras(config_path):
    return load_config(config_path).get("CAMERAS", [])


def camera_number(camera, index):
    value = str(camera.get("id", ""))
    return value if value.isdigit() and int(value) > 0 else str(index)


def camera_key(number):
    return f"camera/{number}"


def external_camera_key(edge_id, number):
    return f"{edge_id}/{camera_key(number)}"


def load_edge_id(path):
    try:
        edge_id = json.loads(path.read_text(encoding="utf-8")).get("edge_id")
    except (OSError, json.JSONDecodeError):
        edge_id = None
    if not isinstance(edge_id, str) or not edge_id.strip():
        raise ValueError(
            "edge_id가 없습니다. 먼저 `python apps/edge_client/agent.py init`을 실행하세요."
        )
    return edge_id


def next_camera_number(cameras):
    numbers = [
        int(camera["id"])
        for camera in cameras
        if str(camera.get("id", "")).isdigit() and int(camera["id"]) > 0
    ]
    return str(max(numbers, default=0) + 1)


def numbered_cameras(cameras):
    reserved = {
        str(camera.get("id"))
        for camera in cameras
        if str(camera.get("id", "")).isdigit() and int(camera["id"]) > 0
    }
    used = set()
    fallback = 1
    for camera in cameras:
        number = str(camera.get("id", ""))
        if not number.isdigit() or int(number) <= 0 or number in used:
            while str(fallback) in used or str(fallback) in reserved:
                fallback += 1
            number = str(fallback)
        used.add(number)
        yield number, camera


def rtsp_endpoint(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    port = f":{parsed.port}" if parsed.port else ""
    return urlunsplit((parsed.scheme, f"{host}{port}", parsed.path, parsed.query, ""))


def print_cameras(config_path):
    cameras = load_cameras(config_path)
    print(f"\n등록된 카메라 (설정위치: {config_path})")
    if not cameras:
        print("  없음")
        return

    for number, camera in numbered_cameras(cameras):
        endpoint = rtsp_endpoint(camera["url"])
        print(
            f"  {number}. {camera_key(number)} | "
            f"{camera.get('name', number)} "
            f"({endpoint}, {camera.get('location') or '-'}, "
            f"intrinsic={'등록됨' if camera.get('intrinsic') else '미등록'})"
        )


def save_camera(config_path, camera):
    config = load_config(config_path)
    cameras = config.setdefault("CAMERAS", [])
    if any(item.get("id") == camera["id"] for item in cameras):
        raise ValueError(f"이미 등록된 카메라 ID입니다: {camera['id']}")
    if any(item.get("name") == camera["name"] for item in cameras):
        raise ValueError(f"이미 등록된 카메라 이름입니다: {camera['name']}")
    if any(
        rtsp_endpoint(item.get("url", "")) == rtsp_endpoint(camera["url"])
        for item in cameras
    ):
        raise ValueError("이미 등록된 RTSP 주소입니다.")
    cameras.append(camera)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    os.chmod(config_path, 0o600)


def save_intrinsic(config_path, camera_number_value, intrinsic):
    config = load_config(config_path)
    for number, camera in numbered_cameras(config.get("CAMERAS", [])):
        if number == camera_number_value:
            camera["intrinsic"] = intrinsic
            break
    else:
        raise ValueError(f"등록되지 않은 카메라 번호입니다: {camera_number_value}")
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )
    os.chmod(config_path, 0o600)


def _float_values(prompt, count):
    values = [float(value) for value in input(prompt).replace(",", " ").split()]
    if len(values) != count:
        raise ValueError(f"{count}개의 숫자를 입력해주세요.")
    return values


def input_intrinsic():
    fx, fy, cx, cy = _float_values(
        "fx fy cx cy (예: 920.1 918.7 960 540): ", 4
    )
    coefficients = [
        float(value)
        for value in input(
            "왜곡계수 k1 k2 p1 p2 [k3 ...] (예: -0.12 0.03 0.001 0 0): "
        ).replace(",", " ").split()
    ]
    if len(coefficients) not in {4, 5, 8, 12, 14}:
        raise ValueError("왜곡계수는 OpenCV 순서로 4, 5, 8, 12, 14개여야 합니다.")
    width, height = (int(value) for value in _float_values("영상 크기 width height: ", 2))
    if min(fx, fy, width, height) <= 0:
        raise ValueError("초점거리와 영상 크기는 양수여야 합니다.")
    return {
        "camera_matrix": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "distortion_coefficients": coefficients,
        "image_size": [width, height],
        "method": "manual",
        "rms_error": None,
    }


def measure_intrinsic(url, columns=9, rows=6, square_size_m=0.025, views=15):
    """Interactively collect checkerboard views and run OpenCV calibration."""
    import cv2
    import numpy as np

    if columns < 2 or rows < 2 or square_size_m <= 0 or views < 3:
        raise ValueError("체커보드 크기/간격과 촬영 수를 확인해주세요.")
    pattern = (columns, rows)
    object_template = np.zeros((columns * rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2)
    object_template *= square_size_m
    object_points = []
    image_points = []
    image_size = None
    capture = cv2.VideoCapture(url)
    if not capture.isOpened():
        raise RuntimeError("RTSP 스트림을 열 수 없습니다.")
    print("체커보드를 여러 각도로 보여주세요. Space: 채택, q/Esc: 취소")
    try:
        while len(image_points) < views:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("RTSP 프레임을 읽을 수 없습니다.")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern)
            preview = frame.copy()
            if found:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
                )
                cv2.drawChessboardCorners(preview, pattern, corners, found)
            cv2.putText(
                preview, f"views {len(image_points)}/{views}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
            )
            cv2.imshow("intrinsic calibration", preview)
            key = cv2.waitKey(1) & 0xFF
            if key in {27, ord("q")}:
                raise RuntimeError("측정을 취소했습니다.")
            if key == ord(" ") and found:
                object_points.append(object_template.copy())
                image_points.append(corners)
                image_size = (gray.shape[1], gray.shape[0])
    finally:
        capture.release()
        cv2.destroyAllWindows()
    rms, matrix, distortion, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    if not np.isfinite(rms) or not np.isfinite(matrix).all() or not np.isfinite(distortion).all():
        raise RuntimeError("유효한 intrinsic 결과를 계산하지 못했습니다.")
    return {
        "camera_matrix": matrix.tolist(),
        "distortion_coefficients": distortion.reshape(-1).tolist(),
        "image_size": list(image_size),
        "method": "opencv_checkerboard",
        "checkerboard": {
            "inner_corners": [columns, rows],
            "square_size_m": square_size_m,
            "views": views,
        },
        "rms_error": float(rms),
    }


def configure_intrinsic(camera):
    method = input("Intrinsic 등록 [m: 직접입력, c: 체커보드 측정, Enter: 나중에]: ").strip().lower()
    if not method:
        return None
    if method == "m":
        result = input_intrinsic()
    elif method == "c":
        columns, rows = (int(value) for value in input("내부 코너 열 행 [9 6]: ").split() or (9, 6))
        square = float(input("한 칸 크기(m) [0.025]: ").strip() or "0.025")
        views = int(input("촬영 장수 [15]: ").strip() or "15")
        result = measure_intrinsic(camera["url"], columns, rows, square, views)
    else:
        raise ValueError("m, c 또는 Enter 중 하나를 선택해주세요.")
    flattened = [value for row in result["camera_matrix"] for value in row]
    flattened.extend(result["distortion_coefficients"])
    if not all(math.isfinite(float(value)) for value in flattened):
        raise ValueError("Intrinsic 값은 모두 유한한 숫자여야 합니다.")
    result["schema_version"] = 1
    result["calibrated_at"] = time.time()
    return result


def select_cameras(cameras):
    if not cameras:
        raise ValueError("등록된 카메라가 없습니다.")
    answer = input("카메라 번호 (0: 전체): ").strip()
    if answer == "0":
        return list(numbered_cameras(cameras))
    selected = dict(numbered_cameras(cameras)).get(answer)
    if selected is None:
        raise ValueError("목록에 있는 카메라 번호를 입력해주세요.")
    return [(answer, selected)]


def print_probe(index, camera):
    number = camera_number(camera, index)
    key = camera_key(number)
    try:
        stream = probe_rtsp(camera["url"])
        print(
            f"{key}: {stream['codec']}, "
            f"{stream['width']}x{stream['height']}, {stream['fps']} FPS"
        )
        return True
    except RuntimeError as exc:
        print(f"{key}: 연결 실패 - {exc}")
        return False


def show_cameras(selected):
    import cv2

    active = []
    for index, camera in selected:
        if not print_probe(index, camera):
            continue
        capture = cv2.VideoCapture(camera["url"])
        if capture.isOpened():
            active.append((camera_key(camera_number(camera, index)), capture))
        else:
            print(
                f"{camera_key(camera_number(camera, index))}: 스트림을 열 수 없습니다."
            )

    if not active:
        return
    print("창을 선택한 뒤 q 또는 Esc를 누르면 종료합니다.")
    try:
        while True:
            for key, capture in active:
                ok, frame = capture.read()
                if ok:
                    cv2.imshow(key, frame)
            pressed = cv2.waitKey(1) & 0xFF
            if pressed in {27, ord("q")}:
                break
    finally:
        for _, capture in active:
            capture.release()
        cv2.destroyAllWindows()


def capture_images(selected, capture_dir):
    import cv2

    capture_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    for index, camera in selected:
        number = camera_number(camera, index)
        if not print_probe(index, camera):
            continue

        capture = cv2.VideoCapture(camera["url"])
        try:
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            print(f"{camera_key(number)}: 캡처 실패")
            continue

        path = capture_dir / f"camera_{number}_{timestamp}.jpg"
        if cv2.imwrite(str(path), frame):
            print(f"저장 완료: {path}")
        else:
            print(f"{camera_key(number)}: 파일 저장 실패")


def connected_cameras(cameras):
    connected = []
    print("\n카메라 연결 확인 중...")
    for number, camera in numbered_cameras(cameras):
        if print_probe(number, camera):
            connected.append((number, camera))

    print("\n현재 연결 가능한 카메라")
    if not connected:
        print("  없음")
        return []
    for number, camera in connected:
        print(
            f"  {number}. {camera_key(number)} | "
            f"{camera.get('name', number)} ({rtsp_endpoint(camera['url'])})"
        )
    return connected


def _ffmpeg_record_command(
    ffmpeg, url, output_path, preview_width, preview_height, preview_fps
):
    preview_filter = (
        f"fps={preview_fps},"
        f"scale={preview_width}:{preview_height}:force_original_aspect_ratio=decrease,"
        f"pad={preview_width}:{preview_height}:(ow-iw)/2:(oh-ih)/2"
    )
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-i",
        url,
        "-map",
        "0:v:0",
        "-c:v",
        "copy",
        "-an",
        "-f",
        "matroska",
        str(output_path),
        "-map",
        "0:v:0",
        "-an",
        "-vf",
        preview_filter,
        "-c:v",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-f",
        "rawvideo",
        "pipe:1",
    ]


def _read_exact(stream, size):
    data = bytearray()
    while len(data) < size:
        chunk = stream.read(size - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def _read_preview_frames(state, width, height):
    import numpy as np

    frame_size = width * height * 3
    while True:
        data = _read_exact(state["process"].stdout, frame_size)
        if data is None:
            break
        frame = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        with state["lock"]:
            state["frame"] = frame


def _read_ffmpeg_errors(state):
    url = state["camera"]["url"]
    endpoint = rtsp_endpoint(url)
    for raw_line in iter(state["process"].stderr.readline, b""):
        line = raw_line.decode("utf-8", errors="replace").strip().replace(url, endpoint)
        if line:
            state["errors"].append(line)


def _start_recording_stream(
    ffmpeg, number, camera, output_path, preview_width, preview_height, preview_fps
):
    command = _ffmpeg_record_command(
        ffmpeg,
        camera["url"],
        output_path,
        preview_width,
        preview_height,
        preview_fps,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=preview_width * preview_height * 3,
    )
    state = {
        "number": number,
        "camera": camera,
        "output_path": output_path,
        "process": process,
        "frame": None,
        "lock": threading.Lock(),
        "errors": deque(maxlen=5),
    }
    state["preview_thread"] = threading.Thread(
        target=_read_preview_frames,
        args=(state, preview_width, preview_height),
        daemon=True,
    )
    state["error_thread"] = threading.Thread(
        target=_read_ffmpeg_errors, args=(state,), daemon=True
    )
    state["preview_thread"].start()
    state["error_thread"].start()
    return state


def _recording_mosaic(states, elapsed, tile_width, tile_height):
    import cv2
    import numpy as np

    columns = math.ceil(math.sqrt(len(states)))
    rows = math.ceil(len(states) / columns)
    header_height = 54
    canvas = np.zeros(
        (header_height + rows * tile_height, columns * tile_width, 3),
        dtype=np.uint8,
    )
    title = f"REC  {int(elapsed)} sec"
    (title_width, _), _ = cv2.getTextSize(
        title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    )
    cv2.putText(
        canvas,
        title,
        ((canvas.shape[1] - title_width) // 2, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    for position, state in enumerate(states):
        row, column = divmod(position, columns)
        y = header_height + row * tile_height
        x = column * tile_width
        with state["lock"]:
            frame = state["frame"]
        if frame is None:
            tile = np.zeros((tile_height, tile_width, 3), dtype=np.uint8)
            status = "WAITING" if state["process"].poll() is None else "DISCONNECTED"
            (status_width, _), _ = cv2.getTextSize(
                status, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.putText(
                tile,
                status,
                ((tile_width - status_width) // 2, tile_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (160, 160, 160),
                2,
                cv2.LINE_AA,
            )
        else:
            tile = frame.copy()
        label = f"Camera {state['number']}"
        cv2.rectangle(tile, (0, 0), (180, 42), (0, 0, 0), -1)
        cv2.putText(
            tile,
            label,
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        canvas[y : y + tile_height, x : x + tile_width] = tile
    return canvas


def _stop_recording_streams(states):
    for state in states:
        process = state["process"]
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    for state in states:
        process = state["process"]
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        state["preview_thread"].join(timeout=2)
        state["error_thread"].join(timeout=2)


def record_cameras(
    cameras, record_dir, preview_width=640, preview_height=360, preview_fps=5
):
    import cv2

    if min(preview_width, preview_height, preview_fps) <= 0:
        raise ValueError("관제 화면 크기와 FPS는 양수여야 합니다.")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("녹화에 필요한 ffmpeg 실행 파일을 찾을 수 없습니다.")

    connected = connected_cameras(cameras)
    if not connected:
        return
    answer = input("\n녹화를 시작할까요? [y/N]: ").strip().lower()
    if answer not in {"y", "yes", "예", "네"}:
        print("녹화를 취소했습니다.")
        return

    session_dir = record_dir / time.strftime("%Y%m%d-%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    states = []
    try:
        for number, camera in connected:
            output_path = session_dir / f"camera_{number}.mkv"
            states.append(
                _start_recording_stream(
                    ffmpeg,
                    number,
                    camera,
                    output_path,
                    preview_width,
                    preview_height,
                    preview_fps,
                )
            )

        started_at = time.monotonic()
        window_name = "Camera recording monitor"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print("녹화를 시작했습니다. 관제 창에서 q 또는 Esc를 누르면 종료합니다.")
        while True:
            elapsed = time.monotonic() - started_at
            cv2.imshow(
                window_name,
                _recording_mosaic(
                    states, elapsed, preview_width, preview_height
                ),
            )
            if cv2.waitKey(20) & 0xFF in {27, ord("q")}:
                break
            if all(state["process"].poll() is not None for state in states):
                print("모든 카메라 연결이 종료되어 녹화를 중지합니다.")
                break
    except cv2.error as exc:
        raise RuntimeError(
            "관제 창을 열 수 없습니다. GUI가 있는 환경에서 실행해주세요."
        ) from exc
    except KeyboardInterrupt:
        print("\n녹화 중지 요청을 받았습니다.")
    finally:
        _stop_recording_streams(states)
        cv2.destroyAllWindows()

    for state in states:
        output_path = state["output_path"]
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"저장 완료: {output_path}")
        else:
            detail = state["errors"][-1] if state["errors"] else "원인 확인 불가"
            print(f"{camera_key(state['number'])}: 녹화 실패 - {detail}")


def main():
    parser = argparse.ArgumentParser(description="엣지 카메라 한 대 등록")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["list", "show", "capture", "intrinsic", "record"],
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--edge-id-file", type=Path, default=DEFAULT_EDGE_ID_FILE)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    parser.add_argument("--record-dir", type=Path, default=DEFAULT_RECORD_DIR)
    parser.add_argument("--preview-width", type=int, default=640)
    parser.add_argument("--preview-height", type=int, default=360)
    parser.add_argument("--preview-fps", type=int, default=5)
    args = parser.parse_args()

    try:
        load_edge_id(args.edge_id_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print_cameras(args.config)
    if args.command == "list":
        return
    if args.command == "record":
        try:
            record_cameras(
                load_cameras(args.config),
                args.record_dir,
                args.preview_width,
                args.preview_height,
                args.preview_fps,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"녹화 실패: {exc}") from exc
        return
    if args.command in {"show", "capture", "intrinsic"}:
        try:
            selected = select_cameras(load_cameras(args.config))
        except ValueError as exc:
            raise SystemExit(f"실행 실패: {exc}") from exc
        if args.command == "show":
            show_cameras(selected)
        elif args.command == "capture":
            capture_images(selected, args.capture_dir)
        else:
            if len(selected) != 1:
                raise SystemExit("Intrinsic 측정은 카메라 한 대씩 선택해주세요.")
            number, camera = selected[0]
            try:
                intrinsic = configure_intrinsic(camera)
                if intrinsic is None:
                    raise SystemExit("Intrinsic 등록을 취소했습니다.")
                save_intrinsic(args.config, number, intrinsic)
            except (RuntimeError, ValueError, OSError, yaml.YAMLError) as exc:
                raise SystemExit(f"Intrinsic 등록 실패: {exc}") from exc
            print(f"Intrinsic 저장 완료: {camera_key(number)}")
        return

    print("\n엣지 카메라 등록")
    default_number = next_camera_number(load_cameras(args.config))
    camera_id = input(f"카메라 번호 [{default_number}]: ").strip() or default_number
    if not camera_id.isdigit() or int(camera_id) <= 0:
        raise SystemExit("카메라 번호는 1 이상의 정수여야 합니다.")
    camera_id = str(int(camera_id))
    name = input(f"카메라 이름 [{camera_key(camera_id)}]: ").strip() or camera_id
    raw_url = input(
        "RTSP 주소 (계정 제외, 예: rtsp://192.168.0.10:554/stream1): "
    ).strip()
    username = input("사용자 이름 (없으면 Enter): ").strip()
    password = getpass("비밀번호: ") if username else ""
    location = input("설치 위치 (선택): ").strip()
    twin_id = input("Twin ID (선택): ").strip()

    try:
        url = build_rtsp_url(raw_url, username, password)
        print("RTSP 연결 확인 중...")
        try:
            stream = probe_rtsp(url)
            print(
                f"연결 성공: {stream['codec']}, "
                f"{stream['width']}x{stream['height']}, {stream['fps']} FPS"
            )
        except RuntimeError as exc:
            print(f"연결 실패: {exc}")
            answer = input("그래도 카메라 정보를 저장하시겠습니까? [y/N]: ")
            if answer.strip().lower() not in {"y", "yes", "예", "네"}:
                raise SystemExit("등록을 취소했습니다.") from exc

        camera = {
                "id": camera_id,
                "name": name,
                "url": url,
                "location": location or None,
                "twin_id": twin_id or None,
            }
        intrinsic = configure_intrinsic(camera)
        if intrinsic is not None:
            camera["intrinsic"] = intrinsic
        save_camera(args.config, camera)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"등록 실패: {exc}") from exc

    print(f"등록 완료: {name} ({camera_key(camera_id)})")
    print(f"로컬 설정: {args.config}")


if __name__ == "__main__":
    main()
