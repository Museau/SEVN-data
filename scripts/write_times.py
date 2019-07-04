"""Writes timestamps file for ORBSLAM2"""

import argparse

parser = argparse.ArgumentParser("Writes timestamp file for ORBSLAM2")
parser.add_argument('fps', type=float, help='fps')
parser.add_argument('file_len', type=int, help='number of images')
args = parser.parse_args()
fps = args.fps
file_len = file_len

file = open("times.txt","a")
for i in range(file_len): 
    out_fname = "{:06f}".format((i+1)/fps)
    file.write(out_fname+"\n")
file.close()
