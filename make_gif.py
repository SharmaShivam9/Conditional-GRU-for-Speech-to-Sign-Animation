import numpy as np
import imageio.v2 as imageio
import cv2
import os

# --- Configuration ---
POSE_LANDMARKS_COUNT = 4
HAND_LANDMARKS_COUNT = 21
POSE_DIMS = HAND_DIMS = 2
FIXED_HEAD_RADIUS = 0.07
CANVAS_SIZE = 512  # image size (512x512 px)

# Define Skeleton Structure
POSE_LANDMARKS_MAP = {
    'left_shoulder': 0, 'right_shoulder': 1,
    'left_elbow': 2, 'right_elbow': 3,
}

SIMPLE_POSE_CONNECTIONS = [
    (POSE_LANDMARKS_MAP['left_shoulder'], POSE_LANDMARKS_MAP['right_shoulder']),
    (POSE_LANDMARKS_MAP['left_shoulder'], POSE_LANDMARKS_MAP['left_elbow']),
    (POSE_LANDMARKS_MAP['right_shoulder'], POSE_LANDMARKS_MAP['right_elbow'])
]

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)
]

def generate_stick_figure_gif_from_array(Y_pred, output_path, fps=30):
    """
    Fast vectorized GIF generator using OpenCV + ImageIO.
    Y_pred: np.ndarray (T, 92)
    output_path: path to save .gif
    fps: frames per second
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Normalize to 0–1 if not already
    if Y_pred.min() < 0 or Y_pred.max() > 1:
        Y_pred = np.clip(Y_pred, 0, 1)

    # Flatten index splits
    FLAT_POSE_SIZE = POSE_LANDMARKS_COUNT * POSE_DIMS
    FLAT_HAND_SIZE = HAND_LANDMARKS_COUNT * HAND_DIMS

    frames = []
    for frame_landmarks in Y_pred[::3]:  # subsample every 3rd frame
        img = np.ones((CANVAS_SIZE, CANVAS_SIZE, 3), dtype=np.uint8) * 255  # white bg

        # Decode
        pose_end = FLAT_POSE_SIZE
        left_hand_end = pose_end + FLAT_HAND_SIZE
        pose_data = frame_landmarks[:pose_end].reshape(POSE_LANDMARKS_COUNT, 2)
        left_hand = frame_landmarks[pose_end:left_hand_end].reshape(HAND_LANDMARKS_COUNT, 2)
        right_hand = frame_landmarks[left_hand_end:].reshape(HAND_LANDMARKS_COUNT, 2)

        # Scale to image
        def to_px(coords):
            return (coords * CANVAS_SIZE).astype(int)

        pose = to_px(pose_data)
        lh = to_px(left_hand)
        rh = to_px(right_hand)

        # Draw pose lines
        for i, j in SIMPLE_POSE_CONNECTIONS:
            cv2.line(img, tuple(pose[i]), tuple(pose[j]), (0, 0, 255), 3)

        # Draw hand connections
        for i, j in HAND_CONNECTIONS:
            cv2.line(img, tuple(lh[i]), tuple(lh[j]), (255, 0, 0), 1)
            cv2.line(img, tuple(rh[i]), tuple(rh[j]), (0, 255, 0), 1)

        # Draw connecting arms
        cv2.line(img, tuple(pose[2]), tuple(lh[0]), (0, 0, 255), 2)
        cv2.line(img, tuple(pose[3]), tuple(rh[0]), (0, 0, 255), 2)

        # Draw head (circle)
        head_center = ((pose[0] + pose[1]) // 2)
        head_center[1] -= int(FIXED_HEAD_RADIUS * CANVAS_SIZE)
        cv2.circle(img, tuple(head_center), int(FIXED_HEAD_RADIUS * CANVAS_SIZE), (0, 0, 255), 2)

        frames.append(img)

    # Save to GIF
    duration = 1000 / fps  # milliseconds per frame
    imageio.mimsave(output_path, frames, duration=duration)
    print(f" Saved fast GIF: {output_path} ({len(frames)} frames, {fps} fps)")

