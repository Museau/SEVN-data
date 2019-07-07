import csv
import os
import sys
import quaternion
import argparse
import struct
import numpy as np
import pandas as pd
import time
import utils
from shutil import copyfile
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

# python filter_panos.py --coords_file "../poses/coords.csv" --output_file "data/run_1/processed/pos_ang" --pano_path "/Users/martinweiss/code/academic/hyrule-gym/data/data/run_1/panos/"


parser = argparse.ArgumentParser(description='Filter some coords.')
parser.add_argument('--coords_file', type=str, help='a file containing the coords')
parser.add_argument('--output_file', type=str, help='a file where we write the "pos_ang" numpy array')
parser.add_argument('--pano_src', type=str, help='source location for panos')
parser.add_argument('--pano_dst', type=str, help='dest location for panos')
args = parser.parse_args()


def filter_poses(poses):
    to_filter = set()
    for i, node1 in tqdm(poses.iterrows(), leave=False, total=poses.shape[0], desc="filtering nodes"):
        for j, node2 in utils.find_nearby_nodes(poses, node1, 0.5).iterrows():
            if i == j or j in to_filter or i in to_filter:
                continue
            to_filter.add(j)
    return poses[~poses.index.isin(to_filter)]


def save_poses(filename, poses):
    np.save(filename, poses.values)


# Filter the poses
poses = pd.read_csv(args.coords_file, delimiter=",")
poses.columns = ['index', 'timestamp', 'x', 'y', 'z', 'angle']
ROT_MAT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
filtered_poses = filter_poses(poses)
save_poses(args.output_file, filtered_poses)
print("num poses filtered: " + str(len(poses) - len(filtered_poses)))
print("num poses remaining: " + str(len(filtered_poses)))

# Write an image of the filtered poses
x_f = [pose.x for idx, pose in filtered_poses.iterrows()]
y_f = [pose.y for idx, pose in filtered_poses.iterrows()]
plt.scatter(x_f, y_f)
plt.savefig("filtered_poses.png")

# Copy the selected panos from the HD
paths = []
for d in os.listdir(args.pano_src):
    path = args.pano_src + "/" + d
    for f in os.listdir(path):
        paths.append(path + "/" + f)

good_paths = []
nums = [str(int(x.timestamp * 30)).zfill(6) for idx, x in filtered_poses.iterrows()]
for path in paths:
    for num in nums:
        if num in path:
            good_paths.append(path)

# os.mkdir(args.pano_dst)
for path in tqdm(good_paths, total=len(good_paths)):
    copyfile(path, args.pano_dst + path.split("/")[-1])
