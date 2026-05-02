# Computer Vision – Camera Calibration & Augmented Reality

## Overview

This project implements a complete computer vision pipeline for:

- Camera calibration using a chessboard pattern
- Homography estimation (custom implementation + OpenCV comparison)
- Camera pose recovery (rotation and translation)
- 3D cube projection on images and videos

The objective is to evaluate the accuracy and robustness of classical geometric methods in realistic conditions.

---

## Features

- Camera calibration from image sets and videos
- Custom homography estimation using SVD
- Comparison with OpenCV (`cv2.findHomography`)
- Camera pose estimation from homography
- 3D cube projection
- Quantitative evaluation:
  - Projection error
  - Homography error

---

## Pipeline

1. Detect chessboard corners
2. Estimate camera intrinsic parameters
3. Compute homography:
   - Manual implementation
   - OpenCV reference
4. Recover camera pose (R, t)
5. Project a 3D cube onto the scene
6. Evaluate errors

---

## Project Structure

```text
src/
    calibration.py
    homography.py
    projection.py
    metrics.py
    pipeline.py

data/
    image_set/
    videos/

outputs/
    images/
    videos/
```

---

## Example Outputs

The pipeline produces:

* Augmented images and videos with a projected 3D cube
* Error plots showing projection and homography accuracy

---

## Key Observations

* Good performance under controlled conditions
* Sensitivity to:
  * lighting conditions
  * image resolution
  * lens distortion
* Better stability in videos due to temporal consistency

---

## Future Improvements

* Bundle adjustment for better accuracy
* Robust estimation (RANSAC)
* Temporal filtering (e.g. Kalman filter)
* Improved corner detection
