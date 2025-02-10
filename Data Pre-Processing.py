import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the video
video_path = '1.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_rate = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / frame_rate

print(f"Frame Rate: {frame_rate}")
print(f"Total Frames: {total_frames}")
print(f"Duration: {duration} seconds")

# Initialize variables
frame_count = 0
processed_frames = []

# Define the augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.0,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Calculate the frame interval for a 1-second gap
frame_interval = int(frame_rate)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Select frames with a 1-second gap
    if frame_count % frame_interval == 0:
        # Noise Cancelling (using Gaussian Blur)
        denoised_frame = cv2.GaussianBlur(frame, (5, 5), 0)

        # Frame resizing
        resized_frame = cv2.resize(denoised_frame, (898, 506))  # Resize to 898x506

        # Frame Normalization (normalize pixel values to [0, 1])
        normalized_frame = resized_frame / 255.0

        # Frame Augmentation (using Keras ImageDataGenerator)
        # Reshape frame to (1, 224, 224, 3) for augmentation
        frame_to_augment = np.expand_dims(normalized_frame, axis=0)

        # Apply augmentation
        augmented_frames = datagen.flow(frame_to_augment, batch_size=1)

        # Retrieve the augmented frame
        augmented_frame = next(augmented_frames)[0]

        # Store the processed frame
        processed_frames.append(augmented_frame)

    frame_count += 1

cap.release()

# Convert processed frames to a numpy array
processed_frames = np.array(processed_frames)

# Save or use the processed frames as needed
for i, frame in enumerate(processed_frames):
    cv2.imwrite(f'Pre-Process/processed_frame_{i}.jpg', frame * 255)  # Convert back to [0, 255] range

# Optionally, you can create a video from the processed frames
output_video_path = 'Pre-Process/1.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, 1.0, (898, 506))  # 1.0 fps to match the 1-second gap

for frame in processed_frames:
    out.write((frame * 255).astype(np.uint8))

out.release()

print("Video processing complete.")
