import torch
import os

"""
필요한 인자들
1. 입력 이미지 사이즈
2. resnet 출력 사이즈
3. 배치사이즈: resnet, root_v2v_net 영향
4. num_views: resnet 영향
"""


def export_tensorrt(model, output_dir, cfg, mode="fp16"):
    from torch2trt import torch2trt

    print("Exporting Model to TensorRT engine...")
    batch_size = cfg.BATCH_SIZE

    device = next(model.parameters()).device

    backbone = model.backbone
    root_v2v_net = model.root_net.v2v_net
    pose_v2v_net = model.pose_net.v2v_net
    root_cube_size = tuple(int(value) for value in model.root_net.cube_size)

    backbone_x = torch.ones(
        (batch_size, 3, cfg.NETWORK.IMAGE_SIZE[1], cfg.NETWORK.IMAGE_SIZE[0]),
        device=device,
    )
    root_v2v_net_x = torch.ones(
        (batch_size, 1, *root_cube_size),
        device=device,
    )
    pose_v2v_net_x = torch.ones(
        (
            1,
            cfg.NETWORK.NUM_JOINTS,
            cfg.PICT_STRUCT.CUBE_SIZE[0],
            cfg.PICT_STRUCT.CUBE_SIZE[1],
            cfg.PICT_STRUCT.CUBE_SIZE[2],
        ),
        device=device,
    )

    if mode == "fp16":
        backbone_trt = torch2trt(backbone, [backbone_x], fp16_mode=True, use_onnx=True)
        root_v2v_net_trt = torch2trt(
            root_v2v_net, [root_v2v_net_x], fp16_mode=True, use_onnx=True
        )
        pose_v2v_net_trt = torch2trt(
            pose_v2v_net, [pose_v2v_net_x], fp16_mode=True, use_onnx=True
        )
    if mode == "int8":
        backbone_trt = torch2trt(backbone, [backbone_x], int8_mode=True, use_onnx=True)
        root_v2v_net_trt = torch2trt(
            root_v2v_net, [root_v2v_net_x], int8_mode=True, use_onnx=True
        )
        pose_v2v_net_trt = torch2trt(
            pose_v2v_net, [pose_v2v_net_x], int8_mode=True, use_onnx=True
        )

    os.makedirs(output_dir, exist_ok=True)
    torch.save(backbone_trt.state_dict(), os.path.join(output_dir, "backbone.pth"))
    torch.save(
        root_v2v_net_trt.state_dict(), os.path.join(output_dir, "root_v2v_net.pth")
    )
    torch.save(
        pose_v2v_net_trt.state_dict(), os.path.join(output_dir, "pose_v2v_net.pth")
    )
    print("Exporting Model to TensorRT engine... Done")


def load_tensorrt_model(model, ckpt_dir, cfg, mode="fp16"):
    from torch2trt import TRTModule

    print("Loading TensorRT model...")
    batch_size = cfg.BATCH_SIZE
    num_views = cfg.NUM_VIEWS
    root_shape = "x".join(str(int(value)) for value in model.root_net.cube_size)
    tensorrt_dir = os.path.join(
        ckpt_dir.split(".")[0],
        f"engine_{mode}_{batch_size}_{num_views}_root_{root_shape}",
    )
    if not os.path.exists(tensorrt_dir):
        export_tensorrt(model, tensorrt_dir, cfg, mode)
    backbone = TRTModule()
    root_v2v_net = TRTModule()
    pose_v2v_net = TRTModule()

    backbone.load_state_dict(
        torch.load(os.path.join(tensorrt_dir, "backbone.pth"), weights_only=False)
    )
    root_v2v_net.load_state_dict(
        torch.load(os.path.join(tensorrt_dir, "root_v2v_net.pth"), weights_only=False)
    )
    pose_v2v_net.load_state_dict(
        torch.load(os.path.join(tensorrt_dir, "pose_v2v_net.pth"), weights_only=False)
    )

    model.backbone = backbone
    model.root_net.v2v_net = root_v2v_net
    model.pose_net.v2v_net = pose_v2v_net
    print("Loading TensorRT model... Done")
    return model
