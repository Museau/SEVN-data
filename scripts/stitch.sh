#!/bin/sh
# $1 = input dir
# $2 = output dir
# $3 = frame num

pto_gen -o "$2/pano_$3.pto" -p 2 -f 120 "$1/image_1/$3.png" "$1/image_3/$3.png" "$1/image_5/$3.png" "$1/image_7/$3.png"
cpfind -o "$2/pano_$3.pto" --multirow --celeste "$2/pano_$3.pto"
cpclean -o "$2/pano_$3.pto" "$2/pano_$3.pto"
linefind -o "$2/pano_$3.pto" "$2/pano_$3.pto"
autooptimiser -a -m -l -s -o "$2/pano_$3.pto" "$2/pano_$3.pto"
pano_modify --canvas=AUTO --crop=AUTO -o "$2/pano_$3.pto" "$2/pano_$3.pto"
hugin_executor --stitching --prefix="$2/pano_$3" "$2/pano_$3.pto"
convert "$2/pano_$3.tif" "$2/pano_$3.png"
rm "$2/pano_$3.tif"
rm "$2/pano_$3.pto"

