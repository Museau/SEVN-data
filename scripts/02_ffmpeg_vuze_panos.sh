#!/bin/sh

# $1 = video path
# $2 = destination folder

mkdir -p $2
ffmpeg -i $1 -r 30 -map 0:0 "$2/pano_%06d.png"








