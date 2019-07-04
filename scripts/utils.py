import quaternion
import numpy as np
import pandas as pd
ROT_MAT = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

def find_nearby_nodes(coords_df, coords, radius):
    return coords_df[(coords_df.x > coords.x - radius) & (coords_df.x < coords.x + radius) & (coords_df.y > coords.y - radius) & (coords_df.y < coords.y + radius)]

def find_angle(ref_q, q):
    rqvec = np.quaternion(ref_q[0], ref_q[1], ref_q[2], ref_q[3])
    qvec = np.quaternion(q[0], q[1], q[2], q[3])
    new_q = rqvec.inverse()*qvec
    ang = np.dot(np.array([0,1,0]), quaternion.as_rotation_vector(new_q))*180/np.pi
    angle = ang % 360
    return angle

def read_cameras_text(path):
    poses = []
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                timestamp = float(elems[0])
                t = np.array([float(elems[1]), float(elems[2]), float(elems[3])])
                t = np.dot(ROT_MAT, t)
                t[2] = 0.0
                q = np.array([float(elems[4]), float(elems[5]), float(elems[6]), float(elems[7])])
                poses.append([timestamp, t, q])
    ref_pose = poses[0]
    angles = []
    for pose in poses:
        angles.append(find_angle(ref_pose[-1], pose[-1]))
    poses = np.array(poses)
    poses[:, -1] = angles
    return poses
