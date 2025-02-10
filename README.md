Data Pre-Processing for Video Frames

This repository contains a Python script for preprocessing video frames using OpenCV and TensorFlow's Keras ImageDataGenerator. The script performs the following tasks:

    Frame Extraction: Captures frames from a video at 1-second intervals.
    Noise Reduction: Applies Gaussian blur to remove noise.
    Resizing: Resizes frames to 898x506 pixels.
    Normalization: Normalizes pixel values to the range [0,1].
    Data Augmentation: Enhances frames using rotation, shifts, shear, and flips.
    Saving Processed Frames: Stores augmented frames as images.
    Video Reconstruction: Converts processed frames back into a video.

Dependencies

    OpenCV
    NumPy
    TensorFlow/Keras

Usage

Place your input video (1.mp4) in the root directory and run:

    python Data Pre-Processing.py
