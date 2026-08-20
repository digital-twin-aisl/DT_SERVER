# ruff: noqa: E402

import os
import sys
import io
import atexit

import cv2
import numpy as np

if "bool" not in np.__dict__:
    np.bool = np.bool_

import torch
from PIL import Image

import onnx
import onnxoptimizer
from torch.onnx import OperatorExportTypes
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoprimaryctx  # noqa: F401 - initializes CUDA's primary context

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fastreid_dir = os.path.join(base_dir, "reid", "fast-reid")
sys.path.append(fastreid_dir)

from fastreid.config import get_cfg
from fastreid.modeling import build_model
from fastreid.utils.checkpoint import Checkpointer
from fastreid.data.transforms import build_transforms
from fastreid.utils.file_io import PathManager

cuda.init()
_dev = cuda.Device(0)
_ctx = _dev.retain_primary_context()
_ctx.push()


def _cleanup_cuda_context():
    """Pop the context owned by this module before PyCUDA cleanup."""
    try:
        if cuda.Context.get_current() is not None:
            cuda.Context.pop()
    except Exception:
        pass


atexit.register(_cleanup_cuda_context)

alpha = 1.0


# ======================================================================
# 기본 PyTorch 기반 Feature Extractor
# ======================================================================

def build_feature_extractor():
    """
    FastReID PyTorch 모델과 전처리 transform을 생성한다.
    """
    cfg_path = os.path.join(fastreid_dir, "configs", "Market1501", "bagtricks_R50-ibn.yml")
    weights_path = os.path.join(fastreid_dir, "weights", "market_bot_R50-ibn.pth")

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    transform = build_transforms(cfg, is_train=False)
    return model, transform


@torch.no_grad()
def extract_features_from_persons(model, transform, person_list):
    """
    PyTorch FastReID 모델을 사용해 person_list에서 feature를 추출한다.
    Procrustes distance 기반 가중치를 feature에 곱해 반환한다.
    """
    features = []
    all_keypoints = [p["keypoints"] for p in person_list]

    for i, person in enumerate(person_list):
        crop_img = person["crop"]
        ref_kp = person["keypoints"]

        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(crop_pil).unsqueeze(0).to(next(model.parameters()).device)

        output = model(img_tensor)
        feat = output.cpu().numpy().squeeze(0)

        dists = [
            procrustes_distance(ref_kp, other_kp)
            for j, other_kp in enumerate(all_keypoints)
            if j != i
        ]
        dists = [d for d in dists if not np.isnan(d)]

        weight = 1 + alpha * np.mean(dists) if dists else 1.0
        feat_weighted = feat * weight

        features.append(
            {
                "edge_id": person["edge_id"],
                "cam": person["cam"],
                "frame": person["frame"],
                "person_idx": person["person_idx"],
                "bbox": [float(x) for x in person["bbox"]],
                "feature": feat_weighted.astype(np.float32, copy=False),
                "weight": float(weight),
            }
        )

    return features


def procrustes_distance(A, B):
    """
    두 keypoint 세트 A, B 사이의 Procrustes distance를 계산한다.
    실패 시 np.nan을 반환한다.
    """
    try:
        A_mean, B_mean = A.mean(axis=0), B.mean(axis=0)
        A_centered, B_centered = A - A_mean, B - B_mean

        norm_A, norm_B = np.linalg.norm(A_centered), np.linalg.norm(B_centered)
        A_scaled, B_scaled = A_centered / norm_A, B_centered / norm_B

        U, _, Vt = np.linalg.svd(B_scaled.T @ A_scaled)
        R = U @ Vt
        A_aligned = A_scaled @ R.T

        return np.sqrt(((A_aligned - B_scaled) ** 2).sum())
    except Exception:
        return np.nan


# ======================================================================
# TensorRT 기반 Feature Extractor 로딩 및 추론
# ======================================================================

def build_trt_feature_extractor():
    """
    TensorRT FastReID 엔진과 전처리 transform을 생성한다.
    반환값:
        engine, context, input, output, bindings, stream, transform
    """
    cfg_path = os.path.join(fastreid_dir, "configs", "Market1501", "bagtricks_R50-ibn.yml")
    weights_path = os.path.join(fastreid_dir, "weights", "market_bot_R50-ibn.engine")

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)

    engine, context, input, output, bindings, stream = _load_engine(weights_path)
    transform = build_transforms(cfg, is_train=False)

    return engine, context, input, output, bindings, stream, transform


@torch.no_grad()
def extract_features_from_persons_trt(model, context, input, output, bindings, stream, transform, person_list):
    """
    TensorRT FastReID 엔진을 사용해 person_list에서 feature를 추출한다.
    Procrustes distance 기반 가중치를 feature에 곱해 반환한다.
    model 인자는 TensorRT engine이며, 로직상 사용하지 않으므로 그대로 둔다.
    """
    features = []
    all_keypoints = [p["keypoints"] for p in person_list]

    for i, person in enumerate(person_list):
        crop_img = person["crop"]
        ref_kp = person["keypoints"]

        crop_pil = Image.fromarray(cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB))
        img_tensor = transform(crop_pil).unsqueeze(0)

        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.contiguous().cpu().numpy().astype(np.float32)

        output_tensor = infer(context, input, output, bindings, stream, img_tensor)
        feat = output_tensor.numpy().squeeze(0)  

        dists = [
            procrustes_distance(ref_kp, other_kp)
            for j, other_kp in enumerate(all_keypoints)
            if j != i
        ]
        dists = [d for d in dists if not np.isnan(d)]

        weight = 1 + alpha * np.mean(dists) if dists else 1.0
        feat_weighted = feat * weight

        features.append(
            {
                "edge_id": person["edge_id"],
                "cam": person["cam"],
                "frame": person["frame"],
                "person_idx": person["person_idx"],
                "bbox": [float(x) for x in person["bbox"]],
                "feature": feat_weighted.tolist(),
                "weight": float(weight),
            }
        )

    return features


def _load_engine(trt_file):
    """
    TensorRT 엔진 파일을 로드하고 실행에 필요한 버퍼를 준비한다.
    반환:
        engine, context, inputs, outputs, bindings, stream
    """
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
    trt_logger = trt.Logger()

    with open(trt_file, "rb") as f, trt.Runtime(trt_logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()
    inputs, outputs, bindings, stream = allocate_buffers(engine, context)

    return engine, context, inputs, outputs, bindings, stream


class HostDeviceMem:
    """Host와 Device 메모리 쌍을 보관하는 클래스."""

    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine, context, input_shape=(1, 3, 256, 128)):
    """
    TensorRT explicit batch 전용 버퍼 할당.
    모든 입력 바인딩에 대해 shape를 지정하고 host/device 메모리와 bindings를 준비한다.
    """
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()

    # 1) 모든 입력 바인딩의 동적 shape 지정
    for i in range(engine.num_bindings):
        if engine.binding_is_input(i):
            context.set_binding_shape(i, tuple(input_shape))

    # 2) 바인딩별 메모리 할당
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        dtype = trt.nptype(engine.get_binding_dtype(i))
        shape = tuple(context.get_binding_shape(i))

        if any(d < 0 for d in shape):
            raise RuntimeError(f"바인딩 {name}의 shape가 아직 확정되지 않음: {shape}")

        size = int(trt.volume(shape))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)

        bindings.append(int(device_mem))
        hd = HostDeviceMem(host_mem, device_mem)

        if engine.binding_is_input(i):
            inputs.append(hd)
        else:
            outputs.append(hd)

    return inputs, outputs, bindings, stream


def infer(context, input, output, bindings, stream, data):
    """
    TensorRT 엔진으로 추론을 수행한다.
    반환:
        torch.Tensor (CPU) shape: (1, -1)
    """
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    data = np.ascontiguousarray(data, dtype=np.float32)
    data_flat = data.ravel()

    # 입력 복사
    for _inp in input:
        if data_flat.size != _inp.host.size:
            raise RuntimeError(
                f"Input data size {data_flat.size} doesn't match buffer size {_inp.host.size}"
            )
        np.copyto(_inp.host, data_flat)

    # H2D
    for inp in input:
        cuda.memcpy_htod_async(inp.device, inp.host, stream)

    # 모든 바인딩 shape 지정되었는지 확인
    assert context.all_binding_shapes_specified, "Binding shape 미지정 바인딩 존재"

    # 실행
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)

    # D2H
    for out in output:
        cuda.memcpy_dtoh_async(out.host, out.device, stream)

    stream.synchronize()

    output_data = output[0].host.reshape(1, -1).copy()
    return torch.from_numpy(output_data)  # CPU tensor


# ======================================================================
# ONNX → TensorRT 변환 래퍼
# ======================================================================

def export_trt_model():
    """
    기존에 export된 ONNX 모델을 TensorRT 엔진으로 변환한다.
    """
    onnx_file_path = os.path.join(fastreid_dir, "weights", "market_bot_R50-ibn.onnx")
    output_dir = os.path.join(fastreid_dir, "weights")
    engine_file = os.path.join(output_dir, "market_bot_R50-ibn.engine")

    onnx2trt(onnx_file_path, engine_file)


def onnx2trt(
    onnx_file_path,
    save_path,
    log_level="ERROR",
    max_workspace_size=1,
    strict_type_constraints=False,
):
    """
    ONNX 모델을 TensorRT 엔진으로 변환하는 함수.
    TensorRT 7.x / 8.x / 10.x API 차이를 모두 처리한다.
    """
    trt_logger = trt.Logger(getattr(trt.Logger, log_level))
    builder = trt.Builder(trt_logger)

    print(f"Loading ONNX file from path {onnx_file_path}...")
    explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, trt_logger)

    if isinstance(onnx_file_path, str):
        with open(onnx_file_path, "rb") as f:
            print("Beginning ONNX file parsing")
            flag = parser.parse(f.read())
    else:
        flag = parser.parse(onnx_file_path.read())

    if not flag:
        for error_idx in range(parser.num_errors):
            print(parser.get_error(error_idx))

    print("Completed parsing of ONNX file.")

    # Output tensor 재지정 (identity를 거쳐 이름을 부여)
    output_tensors = [network.get_output(i) for i in range(network.num_outputs)]
    for tensor in output_tensors:
        network.unmark_output(tensor)

    for tensor in output_tensors:
        identity_out_tensor = network.add_identity(tensor).get_output(0)
        identity_out_tensor.name = f"identity_{tensor.name}"
        network.mark_output(tensor=identity_out_tensor)

    config = builder.create_builder_config()

    # TensorRT 8.0+ 는 set_memory_pool_limit 사용
    workspace_size = max_workspace_size * (1 << 25)
    try:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    except (AttributeError, TypeError):
        config.max_workspace_size = workspace_size

    assert builder.platform_has_fast_fp16, "not support fp16"

    # FP16 설정 (버전별 분기)
    try:
        config.set_flag(trt.BuilderFlag.FP16)
    except (AttributeError, TypeError):
        builder.fp16_mode = True

    if strict_type_constraints:
        config.set_flag(trt.BuilderFlag.STRICT_TYPES)

    print(f"Building an engine from file {onnx_file_path}; this may take a while...")

    # TensorRT 버전별 엔진 빌드
    try:
        # TensorRT 10.x
        serialized_engine = builder.build_serialized_network(network, config)
        print("Create engine successfully!")
        print(f"Saving TRT engine file to path {save_path}")
        with open(save_path, "wb") as f:
            f.write(serialized_engine)
        print(f"Engine file has already saved to {save_path}!")
    except (AttributeError, TypeError):
        try:
            # TensorRT 8.x
            engine = builder.build_engine(network, config)
            print("Create engine successfully!")
            print(f"Saving TRT engine file to path {save_path}")
            with open(save_path, "wb") as f:
                f.write(engine.serialize())
            print(f"Engine file has already saved to {save_path}!")
        except (AttributeError, TypeError):
            # TensorRT 7.x
            engine = builder.build_cuda_engine(network)
            print("Create engine successfully!")
            print(f"Saving TRT engine file to path {save_path}")
            with open(save_path, "wb") as f:
                f.write(engine.serialize())
            print(f"Engine file has already saved to {save_path}!")


# ======================================================================
# PyTorch → ONNX 변환
# ======================================================================

def export_onnx_model():
    """
    FastReID PyTorch 모델을 ONNX 포맷으로 export한다.
    """
    cfg_path = os.path.join(fastreid_dir, "configs", "Market1501", "bagtricks_R50-ibn.yml")
    weights_path = os.path.join(fastreid_dir, "weights", "market_bot_R50-ibn.pth")
    output_dir = os.path.join(fastreid_dir, "weights")

    cfg = get_cfg()
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.WEIGHTS = weights_path
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(cfg)
    model.eval()
    Checkpointer(model).load(cfg.MODEL.WEIGHTS)

    inputs = torch.randn(
        1, 3, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1]
    ).to(model.device)

    onnx_model = change_onnx_model(model, inputs)

    PathManager.mkdirs(output_dir)
    save_path = os.path.join(output_dir, "market_bot_R50-ibn.onnx")
    onnx.save_model(onnx_model, save_path)


def remove_initializer_from_input(model):
    """
    ONNX 그래프의 initializer를 input 목록에서 제거한다 (IR 버전 4 이상일 때만).
    """
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initializer in graph input")
        return model

    inputs = model.graph.input
    name_to_input = {input.name: input for input in inputs}

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])

    return model


def change_onnx_model(model, inputs):
    """
    PyTorch 모델을 trace/export 하여 ONNX 모델을 생성하고,
    ONNX optimizer를 통해 경량화한 모델을 반환한다.
    """
    assert isinstance(model, torch.nn.Module)

    def _check_eval(module):
        assert not module.training

    model.apply(_check_eval)

    print("Beginning ONNX file converting")

    with torch.no_grad():
        with io.BytesIO() as f:
            torch.onnx.export(
                model,
                inputs,
                f,
                operator_export_type=OperatorExportTypes.ONNX_ATEN_FALLBACK,
            )
            onnx_model = onnx.load_from_string(f.getvalue())

    print("Completed convert of ONNX model")

    # ONNX Optimization
    print("Beginning ONNX model path optimization")
    all_passes = onnxoptimizer.get_available_passes()
    passes = ["extract_constant_to_initializer", "eliminate_unused_initializer", "fuse_bn_into_conv"]
    assert all(p in all_passes for p in passes)
    onnx_model = onnxoptimizer.optimize(onnx_model, passes)
    print("Completed ONNX model path optimization")

    return onnx_model
