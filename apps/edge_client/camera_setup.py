#!/usr/bin/env python3
"""Register one camera in the edge-local YAML configuration."""

import argparse
from getpass import getpass
import json
import os
from pathlib import Path
import time
from urllib.parse import quote, urlsplit, urlunsplit

import yaml


DEFAULT_CONFIG = Path(__file__).parent / "config" / "cameras.local.yaml"
DEFAULT_CAPTURE_DIR = Path(__file__).parent / "data" / "captures"
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
            f"({endpoint}, {camera.get('location') or '-'})"
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


def main():
    parser = argparse.ArgumentParser(description="엣지 카메라 한 대 등록")
    parser.add_argument("command", nargs="?", choices=["list", "show", "capture"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--edge-id-file", type=Path, default=DEFAULT_EDGE_ID_FILE)
    parser.add_argument("--capture-dir", type=Path, default=DEFAULT_CAPTURE_DIR)
    args = parser.parse_args()

    try:
        load_edge_id(args.edge_id_file)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print_cameras(args.config)
    if args.command == "list":
        return
    if args.command in {"show", "capture"}:
        try:
            selected = select_cameras(load_cameras(args.config))
        except ValueError as exc:
            raise SystemExit(f"실행 실패: {exc}") from exc
        if args.command == "show":
            show_cameras(selected)
        else:
            capture_images(selected, args.capture_dir)
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

        save_camera(
            args.config,
            {
                "id": camera_id,
                "name": name,
                "url": url,
                "location": location or None,
                "twin_id": twin_id or None,
            },
        )
    except (OSError, ValueError, yaml.YAMLError) as exc:
        raise SystemExit(f"등록 실패: {exc}") from exc

    print(f"등록 완료: {name} ({camera_key(camera_id)})")
    print(f"로컬 설정: {args.config}")


if __name__ == "__main__":
    main()
