import os
import sys
import struct
import utils
import numpy as np
import pandas as pd
import quaternion
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

parser = argparse.ArgumentParser("Extracts camera positions from ORBSLAM2 txt file")
parser.add_argument('input_path', type=str, help='Input path')
parser.add_argument('--plot', action='store_true', help='Plot poses')

args = parser.parse_args()


ROT_MAT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
FPS = 30


class Pose():
    def __init__(self, timestamp, t, angle):
        self.timestamp = timestamp
        self.t = t
        self.angle = angle


def stitch(poses1, poses2, scale_x, scale_y, angle, x, y):
    poses2 = rotate_poses(poses2, angle)
    poses2 = scale_poses(poses2, poses1, scale_x, scale_y)
    intersection = poses1[-1]
    delta = intersection.t - poses2[0].t
    reconstruction = poses1
    for pose in poses2:
        pose.t = pose.t + delta + np.array([x, y, 0])
        reconstruction.append(pose)
    return reconstruction

def scale_poses(poses, ref, scale_x, scale_y):
    dist = []
    ref_dist = []
    for idx in range(len(poses)-1):
        dist.append(np.linalg.norm(poses[idx+1].t - poses[idx].t) / (poses[idx+1].timestamp - poses[idx].timestamp))
    for idx in range(len(ref)-1):
        ref_dist.append(np.linalg.norm(ref[idx+1].t - ref[idx].t) / (ref[idx+1].timestamp - ref[idx].timestamp))
    avg_dist = np.average(dist)
    avg_ref_dist = np.average(ref_dist)
    scale_factor = avg_ref_dist / avg_dist

    for pose in poses:
        pose.t[0] = pose.t[0] * scale_factor *scale_x
        pose.t[1] = pose.t[1] * scale_factor *scale_y

    return poses

def rotate_poses(poses, angle):
    rot_point = poses[0].t
    rot_mat = np.array([[np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), 0], \
                        [np.sin(np.deg2rad(angle)), np.cos(np.deg2rad(angle)), 0],  \
                        [0, 0, 1]])
    for pose in poses:
        tvec = pose.t - rot_point
        tvec = np.dot(rot_mat, tvec)
        pose.t = rot_point + tvec
        pose.angle = (pose.angle + angle) % 360

    return poses


def plot_poses(poses, l):
    x = []; y = []; z = [];
    for pose in poses:
        x.append(pose.t[0])
        y.append(pose.t[1])
        z.append(pose.t[2])
    plot_data(x, y, z, l)


def plot_data(x, y, z, l):
    plt.scatter(x[0:l], y[0:l], c='blue')
    plt.scatter(x[l:-1], y[l:-1], c='red')
    plt.axis('equal')
    plt.show()


folder = args.input_path
reconstructions = {}
for idx in range(18):
    reconstructions["{:02d}".format(idx + 1)] = [Pose(*x) for x in utils.read_cameras_text(folder + "{:02d}".format(idx + 1) + '.txt')]
test = stitch(reconstructions['01'], reconstructions['02'], 1.0, 1.0, -5.0, 1.0, 0.5)
test = stitch(test, reconstructions['03'], 1.0, 1.0, -96.0, 0.0, 0.0)
test = stitch(test, reconstructions['04'], 1.1, 1.1, 182.0, -6.0, 0.0)
test = stitch(test, reconstructions['05'], 1.05, 1.05, 87.5, 0.0, 3.0)
test = stitch(test, reconstructions['06'], 1.09, 1.09, -3.0, -1.0, 0.0)
test = stitch(test, reconstructions['07'], 1.0, 1.0, -95.0, 0.0, -1.0)
test = stitch(test, reconstructions['08'], 1.15, 1.0, -7.0, 0.0, -1.0)
test = stitch(test, reconstructions['09'], 1.06, 1.06, -85.5, 0.0, 4.0)
#test = stitch(test, reconstructions['10'], 1.15, 1.15, 180, 0.0, 0.0)
test = stitch(test, reconstructions['11'], 1.0, 1.0, 90, -10.0, 220.0)
test = stitch(test, reconstructions['12'], 1.05, 1.1, 5.0, 5.0, 0.0)
test = stitch(test, reconstructions['13'], 1.0, 1.0, 91.0, 5.0, -12.5)
test = stitch(test, reconstructions['14'], 1.0, 1.0, 87.0, 0.0, 3.0)
test = stitch(test, reconstructions['15'], 1.1, 1.0, -1.0, 0.0, 0.0)
test = stitch(test, reconstructions['16'], 1.0, 1.0, 176.0, 0.0, -1.0)
test = stitch(test, reconstructions['17'], 1.0, 1.0, 181.5, -3.0, -2.0)
test = stitch(test, reconstructions['18'], 1.2, 0.95, 82.0, 0.0, 0.0)

if args.plot:
    plot_poses(test, len(test))

res = []
for pose in test:
    res.append((pose.timestamp, pose.t[0], pose.t[1], pose.t[2], pose.angle))
res = pd.DataFrame(res, columns = ["timestamp", "x", "y", "z", "angle"])
print(args.input_path)
res.to_csv(args.input_path + "coords.csv")
