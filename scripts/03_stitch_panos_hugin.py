"""Stitch panoramas"""

import os
from shutil import copyfile
import subprocess
import argparse

parser = argparse.ArgumentParser("Creates panoramas from vuze pictures")
parser.add_argument('input_path', type=str,
                    help='folder containing all images')
args = parser.parse_args()

print("Path : " + args.input_path)
input_path = args.input_path

if not os.path.exists(input_path + '/../panoramas'):
    os.makedirs(input_path + '/../panoramas')

output_dir = input_path + '/../panoramas'

for fname in os.listdir(input_path + '/image_1/'):
    frame = fname.split('.')[0]
    if not (os.path.exists(output_dir + frame + '.png')):
        try:
            cmd = ['./scripts/stitch.sh', input_path, output_dir, frame]
            print(cmd)
            print(subprocess.check_output(cmd))
        except KeyboardInterrupt:
            break
        except:
            pass