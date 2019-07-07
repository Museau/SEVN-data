# SEVN-data

Data pre-processing for SEVN: Sidewalk Simulator Environment for Visual Navigation. This takes raw 360 degree video as an input. The camera used was the Vuze+.

## Pre-processing steps

1. Extract 30 FPS frames from the raw video, crop them to get each camera's images separately and undistort the front-facing view. (scripts/01_preprocess_for_slam.sh)
2. Stitch equirrectangular video using Vuze's VR Software and extract 30 FPS frames from it. (scripts/02_ffmpeg_vuze_panos.sh) An alternative is to use [Hugin Panorama](http://hugin.sourceforge.net/) (scripts/03_stitch_panos_hugin.py)
3. Feed the undistorted images to [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2) to obtain the camera's pose for each frame.
4. In our case, we split the run into smaller reconstructions and stitch them together. (scripts/04_stitch_reconstructions.py)
5. Filter panoramas. (scripts/filter_panos.py)
6. Construct dataset and load it into the [SEVN-gym environment](https://github.com/mweiss17/SEVN) (scripts/06_dataset.py)
