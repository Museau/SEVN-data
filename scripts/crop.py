"""Crops and rotates Vuze images"""

import os
import cv2
import argparse
import multiprocessing as mp
from multiprocessing import Queue

from datetime import datetime
startTime = datetime.now()

def crop_proc(q):
    while True:
        try:
            fname = q.get(True, 1)
            img = cv2.imread(fname)
            frame_num = fname.split(".")[0].split("_")[-1]
            track_num = 1 if "track_1" in fname else 2
            i = 0
            print("Processing: " + fname)

            for x, y in [(0, 0), (W, 0), (0, H), (W, H)]:
                i += 1
                camera_num = i if track_num == 1 else 4 + i
                if not camera_num == 1:
                     continue

                out_fname = "../" + output_folder + f"/image_{camera_num}/{frame_num}.png"
                if os.path.isfile(out_fname):
                    continue
                crop_img(img, x, y, out_fname)
        except:
            return


def crop_img(img, x, y, out_fname):
    crop_img = img[y:y+H, x:x+W]
    out_img = cv2.rotate(crop_img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    cv2.imwrite(out_fname, out_img)

parser = argparse.ArgumentParser("Crops and rotates Vuze images")
parser.add_argument('input_path', type=str, help='Path to folder containing the images')
parser.add_argument('output_folder', type=str, help='Output folder')
args = parser.parse_args()

print("Path : " + args.input_path)
if not os.path.exists(args.input_path):
    parser.error("Input folder does not exist or does not contain any images")
print(args.input_path)

input_path = args.input_path
output_folder = args.output_folder
os.chdir(input_path)
if not os.path.exists("../" + output_folder):
    os.mkdir("../" + output_folder)
    os.mkdir("../" + output_folder + "/image_1")
    # os.mkdir("../" + output_folder + "/image_2")
    # os.mkdir("../" + output_folder + "/image_3")
    # os.mkdir("../" + output_folder + "/image_4")
    # os.mkdir("../" + output_folder + "/image_5")
    # os.mkdir("../" + output_folder + "/image_6")
    # os.mkdir("../" + output_folder + "/image_7")
    # os.mkdir("../" + output_folder + "/image_8")

W = int(3200 / 2)
H = int(2176 / 2)
total_frames = len(os.listdir())
q = Queue()

num_procs = 8
procs = []

for i in range(num_procs):
    p = mp.Process(target=crop_proc, args=(q,))
    procs.append(p)
    p.start()

fnames = [fname for fname in os.listdir() if fname != ".DS_Store"]
while fnames:
    if q.empty():
        q.put(fnames.pop())

for p in procs:
    p.join()

print(datetime.now() - startTime)
