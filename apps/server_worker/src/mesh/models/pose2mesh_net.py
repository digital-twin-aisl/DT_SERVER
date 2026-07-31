import torch
import torch.nn as nn

from ..core.config import cfg as cfg
from . import meshnet, posenet
from ..utils.coord_utils import world2cam,cam2pixel

class FlatPose2Mesh(nn.Module):
    def __init__(self, num_joint, graph_L):
        super(FlatPose2Mesh, self).__init__()
        self.graph_L = graph_L
        self.num_joint = num_joint
        self.pose_lifter = posenet.get_model(num_joint, hid_dim=4096, num_layer=2, p_dropout=0.5, pretrained=cfg.MODEL.posenet_pretrained)
        # self.pose2mesh = meshnet.get_model(num_joint_input_chan=2 + 3, num_mesh_output_chan=3, graph_L=graph_L)
        self.pose2mesh = meshnet.get_model(num_joint_input_chan=3, num_mesh_output_chan=3, graph_L=graph_L)

    def forward(self, pose3d,cams):
        mesh_result=[]
        for zone_idx, data in enumerate(pose3d):
            input_data=[]
            if data[:,:,3] is not -1:
                data=data[:,:,:3]
                data=world2cam(data,cams['R'],cams['t'])
                data=cam2pixel(data,cams['focal'],cams['princpt'])
                data = data.reshape(-1, self.num_joint, 3)
                data = data.detach() / 1000
                input_data.append(data)
            cam_mesh = self.pose2mesh(input_data)
            mesh_result.append(cam_mesh)



        return mesh_result, pose3d


def get_model(num_joint, graph_L):
    model = FlatPose2Mesh(num_joint, graph_L)

    return model


