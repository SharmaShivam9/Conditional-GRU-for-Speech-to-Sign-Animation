import sys, os
sys.stderr = open(os.devnull, 'w')  # temporarily silence stderr
import mediapipe as mp
sys.stderr = sys.__stderr__          # restore normal stderr

import mediapipe as mp
import cv2
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import os
import sys



# --- Standardization Constants ---
TARGET_SHOULDER_WIDTH = 0.2 
TARGET_CENTER_X = 0.5
TARGET_CENTER_Y = 0.55 

# --- Landmark Indices (Hips are INCLUDED for processing) ---
ESSENTIAL_POSE_INDICES = [
    11, 12, # Shoulders
    13, 14, # Elbows
    23, 24  # Hips
]
# New indices for the 6 landmarks we extracted
POSE_LANDMARKS_MAP = {
    'left_shoulder': 0, 'right_shoulder': 1,
    'left_elbow': 2, 'right_elbow': 3,
    'left_hip': 4, 'right_hip': 5,
}

# --- Column/Shape Indices for 3D data (for processing) ---
POSE_DIMS_3D = 3  # x, y, z (Visibility REMOVED)
HAND_DIMS_3D = 3  # x, y, z
POSE_LANDMARKS_COUNT_3D = 6 # 6 landmarks
HAND_LANDMARKS_COUNT_3D = 21

FLAT_POSE_SIZE_3D = POSE_LANDMARKS_COUNT_3D * POSE_DIMS_3D        # 6 * 3 = 18
FLAT_HAND_SIZE_3D = HAND_LANDMARKS_COUNT_3D * HAND_DIMS_3D        # 21 * 3 = 63
# Total 3D features = 18 + 63 + 63 = 144

# --- Column/Shape Indices for 2D data (for saving) ---
POSE_DIMS_2D = 2  # x, y (Visibility REMOVED)
HAND_DIMS_2D = 2  # x, y
POSE_LANDMARKS_COUNT_2D = 4 # MODIFIED: We only save 4 landmarks
FLAT_POSE_SIZE_2D = POSE_LANDMARKS_COUNT_2D * POSE_DIMS_2D        # 4 * 2 = 8
FLAT_HAND_SIZE_2D = HAND_LANDMARKS_COUNT_3D * HAND_DIMS_2D        # 21 * 2 = 42
# Total 2D features = 8 + 42 + 42 = 92

def video_to_landmarks_npy(video_path, output_path, detection_conf=0.5, tracking_conf=0.3):
    """
    Processes a video, extracts 3D data (6-point pose, no visibility), 
    standardizes in 3D, then saves the final result as a 2D .npy file
    (4-point pose, no hips).
    """
    
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=detection_conf,
        min_tracking_confidence=tracking_conf
    )
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    all_landmarks_list = []

    print(f"Processing (3D, 6-Point Pose, No Vis): {os.path.basename(video_path)}")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = holistic.process(image_rgb)
        
        # Extract 3D data
        pose = np.full(FLAT_POSE_SIZE_3D, np.nan)
        if results.pose_landmarks:
            pose_data = []
            all_pose_landmarks = results.pose_landmarks.landmark
            for idx in ESSENTIAL_POSE_INDICES:
                lm = all_pose_landmarks[idx]
                pose_data.extend([lm.x, lm.y, lm.z])
            pose = np.array(pose_data)
        
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.full(FLAT_HAND_SIZE_3D, np.nan)
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.full(FLAT_HAND_SIZE_3D, np.nan)
        
        frame_landmarks = np.concatenate([pose, left_hand, right_hand])
        all_landmarks_list.append(frame_landmarks)

    cap.release()
    holistic.close()
    
    if not all_landmarks_list:
        print(f"Warning: No frames processed for {video_path}.")
        return

    # --- 1. Interpolate (on 3D data) ---
    df = pd.DataFrame(all_landmarks_list)
    df.interpolate(method='linear', limit_direction='both', axis=0, inplace=True)
    df.fillna(0.0, inplace=True) 
    all_landmarks_interpolated = df.to_numpy()

    # --- 2. Smooth (on 3D data) ---
    all_landmarks_smooth = savgol_filter(all_landmarks_interpolated, window_length=5, polyorder=2, axis=0)
    
    # --- 3. Standardize (in 3D) ---
    print(f"Standardizing 3D data for {os.path.basename(video_path)}...")
    
    shoulder_widths = []
    pose_end_idx_3d = FLAT_POSE_SIZE_3D
    for frame_landmarks in all_landmarks_smooth:
        pose_data_flat = frame_landmarks[:pose_end_idx_3d].reshape(POSE_LANDMARKS_COUNT_3D, POSE_DIMS_3D)
        left_shoulder = pose_data_flat[POSE_LANDMARKS_MAP['left_shoulder'], :3] 
        right_shoulder = pose_data_flat[POSE_LANDMARKS_MAP['right_shoulder'], :3]
        current_shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) 
        if current_shoulder_width > 0.01:
            shoulder_widths.append(current_shoulder_width)
    
    if not shoulder_widths: median_shoulder_width = 0.1
    else: median_shoulder_width = np.median(shoulder_widths)

    global_scale_factor = TARGET_SHOULDER_WIDTH / (median_shoulder_width + 1e-6)
    print(f"Stable 3D scale factor set to: {global_scale_factor}")
    
    standardized_data_3d = []
    left_hand_end_idx_3d = pose_end_idx_3d + FLAT_HAND_SIZE_3D
    
    for frame_landmarks in all_landmarks_smooth:
        pose_data_flat = frame_landmarks[:pose_end_idx_3d].reshape(POSE_LANDMARKS_COUNT_3D, POSE_DIMS_3D)
        left_hand_data_flat = frame_landmarks[pose_end_idx_3d:left_hand_end_idx_3d].reshape(HAND_LANDMARKS_COUNT_3D, HAND_DIMS_3D)
        right_hand_data_flat = frame_landmarks[left_hand_end_idx_3d:].reshape(HAND_LANDMARKS_COUNT_3D, HAND_DIMS_3D)
        
        # Use HIP CENTER as anchor
        left_hip = pose_data_flat[POSE_LANDMARKS_MAP['left_hip'], :3]
        right_hip = pose_data_flat[POSE_LANDMARKS_MAP['right_hip'], :3]
        anchor_center = (left_hip + right_hip) / 2 # 3D vector [x,y,z]

        def transform(point_3d):
            new_point = np.copy(point_3d)
            xyz = point_3d[:3]
            scaled_xyz = (xyz - anchor_center) * global_scale_factor 
            new_point[0] = scaled_xyz[0] + TARGET_CENTER_X
            new_point[1] = scaled_xyz[1] + TARGET_CENTER_Y
            new_point[2] = scaled_xyz[2]
            return new_point

        pose_data = np.apply_along_axis(transform, 1, pose_data_flat)
        left_hand_data = np.apply_along_axis(transform, 1, left_hand_data_flat)
        right_hand_data = np.apply_along_axis(transform, 1, right_hand_data_flat)
        
        standardized_frame = np.concatenate([pose_data.flatten(), left_hand_data.flatten(), right_hand_data.flatten()])
        standardized_data_3d.append(standardized_frame)

    # --- 4. Convert 3D data to 2D for Saving (AND REMOVE HIPS) ---
    print(f"Converting standardized 3D data to 2D and removing hips...")
    
    final_data_array_2D = []
    for frame_landmarks_3d in standardized_data_3d:
        pose_data_3d = frame_landmarks_3d[:FLAT_POSE_SIZE_3D].reshape(POSE_LANDMARKS_COUNT_3D, POSE_DIMS_3D)
        left_hand_data_3d = frame_landmarks_3d[FLAT_POSE_SIZE_3D:left_hand_end_idx_3d].reshape(HAND_LANDMARKS_COUNT_3D, HAND_DIMS_3D)
        right_hand_data_3d = frame_landmarks_3d[left_hand_end_idx_3d:].reshape(HAND_LANDMARKS_COUNT_3D, HAND_DIMS_3D)
        
        # --- MODIFIED: Select only the first 4 pose landmarks (Shoulders, Elbows) ---
        pose_data_2d = pose_data_3d[0:4, :2] # x, y. Shape (4, 2)
        left_hand_data_2d = left_hand_data_3d[:, :2] # x, y. Shape (21, 2)
        right_hand_data_2d = right_hand_data_3d[:, :2] # x, y. Shape (21, 2)
        
        frame_data_2d = np.concatenate([
            pose_data_2d.flatten(),
            left_hand_data_2d.flatten(),
            right_hand_data_2d.flatten()
        ])
        final_data_array_2D.append(frame_data_2d)

    # --- 5. Save ---
    final_data_to_save = np.array(final_data_array_2D)
    np.save(output_path, final_data_to_save)
    
    print(f"-> Saved {os.path.basename(output_path)} (Shape: {final_data_to_save.shape})")

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    INPUT_VIDEO_FOLDER = r"D:\Des646\Videos to npy\final_videos"
    OUTPUT_POSE_FOLDER = r"D:\Des646\Videos to npy\pose"

    if not os.path.isdir(INPUT_VIDEO_FOLDER):
        print(f"Error: Input video folder not found: {INPUT_VIDEO_FOLDER}")
        sys.exit(1)
        
    os.makedirs(OUTPUT_POSE_FOLDER, exist_ok=True)
    
    print(f"Starting batch processing...")
    print(f"Input folder: {INPUT_VIDEO_FOLDER}")
    print(f"Output folder: {OUTPUT_POSE_FOLDER}")
    
    video_files_to_process = [
        f for f in os.listdir(INPUT_VIDEO_FOLDER) 
        if f.lower().endswith('.mp4')
    ]
    
    if not video_files_to_process:
        print("No .mp4 videos found to process.")
        sys.exit(0)
        
    print(f"Found {len(video_files_to_process)} video(s) to process...")

    for video_name in video_files_to_process:
        video_path = os.path.join(INPUT_VIDEO_FOLDER, video_name)
        
        file_name_no_ext = os.path.splitext(video_name)[0]
        output_path = os.path.join(OUTPUT_POSE_FOLDER, f"{file_name_no_ext}.npy")
        
        try:
            video_to_landmarks_npy(video_path, output_path)
        except Exception as e:
            print(f"!!! FAILED to process {video_name}: {e}")

    print("\nBatch processing complete.")