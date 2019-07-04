import os
import cv2
import argparse
import yaml
import numpy as np
import multiprocessing as mp
from multiprocessing import Queue

from datetime import datetime
startTime = datetime.now()

parser = argparse.ArgumentParser("Crops and rotates Vuze images")
parser.add_argument('input_path', type=str, help='Path to folder containing the subfolders of images')
args = parser.parse_args()
input_path = args.input_path

def opencv_matrix(loader, node):
    mapping = loader.construct_mapping(node, deep=True)
    mat = np.array(mapping["data"])
    mat.resize(mapping["rows"], mapping["cols"])
    return mat

def undistort_proc(q):
    while True:
        try:
            folder, cam_num, calib, fname = q.get(True, 1)
            out_fname = "../undistorted/" + folder + "/" + fname
            print("Processing: " + folder + "/" + fname)
            print(out_fname)
            if os.path.isfile(out_fname): continue

            img = cv2.imread(folder+ '/' + fname)
            K = calib['K']
            D = calib['D']
            new_K = calib['new_K']

            map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), new_K, DIM, cv2.CV_16SC2)
            undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


            cv2.imwrite(out_fname, undistorted_img)

        except:
            return

yaml.add_constructor(u"tag:yaml.org,2002:opencv-matrix", opencv_matrix)

# loading
with open('/vuze_config/VZP1186200216.yml') as fin:
    c = fin.read()
    # some operator on raw conent of c may be needed
    c = "%YAML 1.1"+os.linesep+"---" + c[len("%YAML:1.0"):] if c.startswith("%YAML:1.0") else c
    result = yaml.load(c, Loader=yaml.Loader)

cam = {}
cam['0'] = result['CamModel_V2_Set']['CAM_0']
cam['1'] = result['CamModel_V2_Set']['CAM_1']
cam['2'] = result['CamModel_V2_Set']['CAM_2']
cam['3'] = result['CamModel_V2_Set']['CAM_3']
cam['4'] = result['CamModel_V2_Set']['CAM_4']
cam['5'] = result['CamModel_V2_Set']['CAM_5']
cam['6'] = result['CamModel_V2_Set']['CAM_6']
cam['7'] = result['CamModel_V2_Set']['CAM_7']

h = 1600
w = 1088
balance = 1
DIM = (1088, 1600)
_img_shape = (1600, 1088)

for k,v in cam.items():
    K = v['K']
    D = np.array(v['DistortionCoeffs'])
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, D, (w, h), 1, (w, h))
    v['D'] = D
    v['new_K'] = new_K

os.chdir(input_path)
q = Queue()
num_procs = 8
procs = []

for i in range(num_procs):
    p = mp.Process(target=undistort_proc, args=(q,))
    procs.append(p)
    p.start()

inputs = []
for folder in os.listdir():
    if not os.path.isdir(folder): continue
    if not (folder == 'image_1'): continue
    cam_num = folder.split('_')[-1]
    calib = cam[str(int(cam_num) - 1)]
    if not os.path.isdir("../undistorted"): os.mkdir("../undistorted")
    if not os.path.exists("../undistorted/" + folder): os.mkdir("../undistorted/" + folder)

    for fname in os.listdir(folder):
        if os.path.isdir(folder + "/" + fname): continue
        inputs.append((folder, cam_num, calib, fname))

while inputs:
    if q.empty():
        q.put(inputs.pop())

for p in procs:
    p.join()

print(datetime.now() - startTime)
