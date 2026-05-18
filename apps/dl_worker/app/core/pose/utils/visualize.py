# This file is part of DT_SERVER.
# 
# DT_SERVER is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or
# (at your option) any later version.
# 
# DT_SERVER is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with DT_SERVER; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA

import pickle
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
import os.path as osp
import os
from tqdm import tqdm
import cv2

LIMBS = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]

def display_3d(preds=None, output=None):
    

    def update_plot(frame_num, ax, fig, pred):
        ax.clear()
        for pose in pred:
            if pose[2][3] == -1:
                continue
            if min(pose[8][2], pose[14][2]) < 0:
                continue

            x = pose[:, 0]
            y = pose[:, 1]
            z = pose[:, 2]

            ax.scatter(x, y, z, c='r', marker='o')

            for limb in LIMBS:
                joint1, joint2 = limb
                x_values = [pose[joint1][0], pose[joint2][0]]
                y_values = [pose[joint1][1], pose[joint2][1]]
                z_values = [pose[joint1][2], pose[joint2][2]]
                ax.plot(x_values, y_values, z_values, 'ro-')

        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        ax.set_title(f'Frame {frame_num}')
        ax.set_xlim(-2500, 2500)  # x축 범위 설정
        ax.set_ylim(-2500, 2500)  # y축 범위 설정
        ax.set_zlim(0, 2000)  # z축 범위 설정

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    if preds is None:
        try:
            with open(output, 'rb') as f:
                data = pickle.load(f)
            preds = data['preds']
        except FileNotFoundError:
            print("Neither preds nor output was given.")

    if output is None:
        output_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..' 'output')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = osp.dirname(output)

    outfile = osp.join(output_dir, '3d_pose.avi')

    writer = imageio.get_writer(outfile, fps=15)

    for i, pred in enumerate(tqdm(preds)):
        update_plot(i, ax, fig, pred)
        fig.canvas.draw()
        frame = np.array(fig.canvas.renderer.buffer_rgba())
        writer.append_data(frame)

    writer.close()

    print(f'Animation saved as {outfile}')


def display_2d(preds_2d=None, view_idx=1, output=None):
    video_file = osp.join(osp.abspath(osp.dirname(__file__)), '..', '..', 'data_0705', 'hdVideos', f'hd_00_{view_idx:02d}.mp4')
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise IOError(f"Failed to open video file: {video_file}")
    
    if preds_2d is None:
        try:
            with open(output, 'rb') as f:
                data = pickle.load(f)
            preds_2d = data['preds_2d']
        except FileNotFoundError:
            print("Neither preds_2d nor output was given.")
    
    if output is None:
        output_dir = osp.join(osp.dirname(osp.abspath(__file__)), '..', '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = osp.dirname(output)


    outfile = osp.join(output_dir, f'2d_pose_view{view_idx}.avi')

    writer = imageio.get_writer(outfile, fps=15)

    preds_2d = np.array(preds_2d)[:,:,view_idx-1,:,:]

    for i in tqdm(range(len(preds_2d))):
        frame = cap.read()[1]
        for pose in preds_2d[i]:
            if np.all(pose == 0):
                continue
            for joint in pose:
                cv2.circle(frame, (int(joint[0]), int(joint[1])), 5, (0, 0, 255), -1)
            for limb in LIMBS:
                joint1, joint2 = limb
                cv2.line(frame, (int(pose[joint1][0]), int(pose[joint1][1])), (int(pose[joint2][0]), int(pose[joint2][1])), (0, 255, 0), 2)
        writer.append_data(frame)

    cap.release()
    writer.close()
    print(f'Animation saved as {outfile}')
    

if __name__ == '__main__':
    display_3d(None, r"/home/dojan/FOCUS-1/modules/SelfPose3d/output/test_2.pkl")
    # preds_2d = display_2d(None, 2, r"/home/dojan/FOCUS-1/modules/SelfPose3d/output/output.pkl")