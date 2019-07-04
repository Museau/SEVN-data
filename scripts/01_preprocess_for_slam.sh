#!/bin/sh

# $1 = video path
# $2 = destination folder

mkdir -p "$2/pngs"

ffmpeg -i $1 -r 30 -map 0:0 "$2/pngs/track_1_output_%06d.png"
python scripts/crop.py $2/pngs crops
python scripts/undistort.py $2/crops

# rm -rf $2/pngs

# Uncomment to convert the undistorted crops to jpgs 
# mkdir -p "$2/undistorted_jpg/image_1"
# mogrify -format jpg -quality 90 -path $2/undistorted_jpg/image_1  $2/undistorted/image_1/*.png
# rm -rf $2/undistorted
 







