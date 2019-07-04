"""Crops top and bottom margins from images"""

import os
import cv2
import argparse

parser = argparse.ArgumentParser("Crops top and bottom margins")
parser.add_argument('input_path', type=str, help='Path to folder containing the images')
parser.add_argument('new_H', type=int, help='Desired image height')
args = parser.parse_args()

input_path = args.input_path
new_H = args.new_H
os.chdir(input_path)

W = int(1088)
H = int(1600)

for idx, fname in enumerate(os.listdir()):
    if fname == ".DS_Store": continue
    if not fname.split('.')[-1] == 'png': continue

    print('Processing: ' + fname + " --- " + str(idx))
    img = cv2.imread(fname)
    crop_img = img[int(H/2 - new_H/2):int(H/2 + new_H/2), 0:W]
    out_fname = '../crop/' + fname
    cv2.imwrite(out_fname, crop_img)

