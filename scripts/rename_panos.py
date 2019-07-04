import os
import argparse


parser = argparse.ArgumentParser("Renames images")
parser.add_argument('input_path', type=str, help='Path to folder containing the images to rename')
parser.add_argument('offset', type=int, help='Offset to add')
args = parser.parse_args()
input_path = args.input_path
offset = args.offset

os.chdir(input_path)

for fname in os.listdir():
    if os.path.isdir(fname): continue
    if not fname.split('.')[-1] == 'png': continue

    frame = int(fname.split('.')[0].split('_')[-1])
    frame = frame + offset
    out_fname = "pano_" + "{:06d}".format(frame) + '.png'
    os.rename(fname, out_fname)