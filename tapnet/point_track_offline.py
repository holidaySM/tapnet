import cv2
import torch
import numpy as np
import os
import time
import argparse
from tapnet.utils import viz_utils

from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import restore_model_from_jax_checkpoint, tracker_certainty

import torch.nn.functional as F
import glob
from PIL import Image


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
    frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
    frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def sample_coordinates_from_mask(mask: np.ndarray, N: int) -> np.ndarray:
    """
    2D 마스크 배열에서 값이 0.5 이상인 영역의 (y, x) 좌표를
    최대 N개까지 무작위로 샘플링합니다.

    Args:
        mask (np.ndarray): 마스크 값을 포함하는 2D NumPy 배열.
        N (int): 샘플링할 최대 좌표 개수. 음수가 아니어야 합니다.

    Returns:
        np.ndarray: 샘플링된 (y, x) 좌표들을 담고 있는 NumPy 배열.
                    배열의 shape은 (k, 2)이며, 여기서 k는 실제로 샘플링된
                    좌표의 개수 (k = min(N, 유효 좌표 개수))입니다.
                    유효한 좌표가 없거나 N=0이면 shape이 (0, 2)인 빈 배열을 반환합니다.
                    반환되는 배열의 dtype은 정수(int)입니다.

    Raises:
        ValueError: 입력 'mask'가 2D NumPy 배열이 아니거나, 'N'이 음수인 경우.
    """
    # --- 입력 유효성 검사 ---
    if not isinstance(mask, np.ndarray) or mask.ndim != 2:
        raise ValueError("입력 'mask'는 2D NumPy 배열이어야 합니다.")
    if not isinstance(N, int) or N < 0:
        raise ValueError("'N'은 음수가 아닌 정수여야 합니다.")

    # N=0이면 바로 빈 배열 반환
    if N == 0:
        return np.empty((0, 2), dtype=int)

    # --- 좌표 찾기 ---
    # 1. mask 값이 0.5 이상인 위치의 인덱스(y, x 좌표)를 찾습니다.
    candidate_indices = np.where(mask >= 0.5)
    # 2. 찾은 y 좌표와 x 좌표를 묶어 (y, x) 쌍의 배열로 만듭니다.
    all_coords = np.column_stack(candidate_indices)

    num_candidates = all_coords.shape[0]

    # --- 샘플링 ---
    # 3. 유효한 좌표가 없으면 빈 배열을 반환합니다.
    if num_candidates == 0:
        return np.empty((0, 2), dtype=int)

    # 4. 실제로 샘플링할 개수(num_samples)를 결정합니다.
    num_samples = min(N, num_candidates)

    # 5. 유효 좌표 중에서 num_samples 개수만큼 무작위로 비복원 추출합니다.
    sampled_indices = np.random.choice(num_candidates, size=num_samples, replace=False)

    # 6. 뽑힌 인덱스에 해당하는 좌표들을 최종 결과로 선택합니다.
    sampled_coords = all_coords[sampled_indices]

    return sampled_coords


class PointTracker:
    def __init__(self, device="cuda"):
        """
        Initializes the PointTracker for offline video processing.
        """
        print("PointTracker...")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.resized_size = 256  # Model input resolution

        model = TAPNext(image_size=(self.resized_size, self.resized_size))
        model_dir = os.path.join(os.path.dirname(__file__), 'weights')
        fname = 'bootstapnext_ckpt.npz'
        ckpt_path = os.path.join(model_dir, 'bootstapnext_ckpt.npz')
        if not os.path.isfile(ckpt_path):
            import requests
            os.makedirs(model_dir, exist_ok=True)
            url = 'https://storage.googleapis.com/dm-tapnet/tapnext/' + fname
            print(f"Downloading checkpoint from {url} to {ckpt_path}...")
            r = requests.get(url)
            with open(ckpt_path, 'wb') as f:
                f.write(r.content)
            print("Download complete.")

        model = restore_model_from_jax_checkpoint(model, ckpt_path)
        model = model.cuda().eval()
        self.tracker_model = model
        # self.tracker_model = torch.compile(self.tracker_model) # Optional: if you have torch 2.0+

        print("PointTracker initialization complete.")

    def process_video(self, rgb_frames_list, mask_frames_list, num_points_per_frame=4, certainty_radius=8,
                      certainty_threshold=0.5):
        """
        Processes an entire video to track points offline.

        Args:
            rgb_frames_list (list of np.ndarray): List of original RGB frames.
            mask_frames_list (list of np.ndarray): List of original mask frames.
            num_points_per_frame (int): Maximum number of points to sample from each mask.

        Returns:
            tuple: (tracks, visibles, rand_color)
                tracks (np.ndarray): Tracks for all points across all frames.
                                     Shape: (num_frames, num_points, 2) (xy coordinates)
                visibles (np.ndarray): Visibility for all points across all frames.
                                       Shape: (num_frames, num_points, 1)
                rand_color (list): List of colors for visualization, one per tracked point.
        """
        if not rgb_frames_list or not mask_frames_list:
            print("No frames or masks provided.")
            return np.empty((0, 0, 2)), np.empty((0, 0, 1)), []

        num_frames = len(rgb_frames_list)
        if len(mask_frames_list) != num_frames:
            print("Warning: Number of RGB frames and mask frames do not match. Using masks available.")
            # Pad masks if needed, or truncate. For this example, let's assume they match or handle a missing mask.
            # If masks are only for specific frames, we need to adapt the query point generation.
            # For now, let's enforce matching count for simplicity.
            # If mask_frames_list has only one mask (like in the original example), we'll replicate it.
            if len(mask_frames_list) == 1:
                mask_frames_list = mask_frames_list * num_frames
            else:
                raise ValueError(
                    "Number of RGB frames and mask frames must match or provide a single mask for all frames.")

        h_org, w_org = rgb_frames_list[0].shape[:2]
        all_resized_rgb_frames = []
        all_query_points_list = []  # Stores (t, y, x) for all sampled points

        # 1. Preprocess frames and collect all query points
        print("Preparing video frames and sampling query points...")
        for frame_idx in range(num_frames):
            rgb_frame = rgb_frames_list[frame_idx]
            mask_frame = mask_frames_list[frame_idx]

            rgb_resized = cv2.resize(rgb_frame, (self.resized_size, self.resized_size), interpolation=cv2.INTER_LINEAR)
            all_resized_rgb_frames.append(torch.from_numpy(rgb_resized).cuda())

            # Resize mask to model's input resolution for point sampling
            mask_resized = cv2.resize(mask_frame.astype(np.uint8), (self.resized_size, self.resized_size),
                                      interpolation=cv2.INTER_NEAREST)

            # Sample points from the current mask
            # sample_coordinates_from_mask returns (y, x)
            sampled_yx_coords = sample_coordinates_from_mask(mask_resized, N=num_points_per_frame)

            if sampled_yx_coords.shape[0] > 0:
                # Add time dimension (frame_idx) to sampled points
                # The query_points format for TAPNext is (t, y, x)
                time_coords = np.full((sampled_yx_coords.shape[0], 1), fill_value=frame_idx, dtype=np.int32)
                # Concatenate (t, y, x)
                frame_query_points = np.concatenate([time_coords, sampled_yx_coords], axis=-1)
                all_query_points_list.append(frame_query_points)

        if not all_query_points_list:
            print("No valid query points found across all masks.")
            return np.empty((num_frames, 0, 2)), np.empty((num_frames, 0, 1)), []

        # Combine all query points into a single tensor
        all_query_points = np.concatenate(all_query_points_list, axis=0)  # Shape: (Total_N, 3) where 3 is (t, y, x)
        all_query_points_torch = torch.from_numpy(all_query_points).cuda().float()  # Must be float for model input
        perm = torch.randperm(all_query_points_torch.size(0))[:64]
        all_query_points_torch = all_query_points_torch[perm]

        # Prepare the full video tensor for model input
        # Convert list of [H, W, C] to [1, T, H, W, C]
        video_tensor = torch.stack(all_resized_rgb_frames, dim=0).unsqueeze(0)  # [1, T, H, W, C]

        print(f"Total frames: {num_frames}, Total query points: {all_query_points.shape[0]}")
        print("Starting TAPNext inference...")
        s_time = time.time()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            # Model output: tracks [1,T,N,2], track_logits [1,T,N,512], raw_visible_logits [1,T,N,1]
            tracks, track_logits, raw_visible_logits, _ = self.tracker_model(
                # <--- 2. Rename 'visibles' to 'raw_visible_logits' for clarity
                video=preprocess_frames(video_tensor),
                query_points=all_query_points_torch.unsqueeze(0)
            )
        print(f"TAPNext inference completed in {time.time() - s_time:.2f} seconds.")

        # tracks: [1, T, N, 2]
        # track_logits: [1, T, N, 512]
        # raw_visible_logits: [1, T, N, 1]

        # 3. Prepare for tracker_certainty: needs [B, Q, T, D]
        #    Model output is [B, T, Q, D], so permute T and Q (dim 1 and 2)
        #    Here N is total_num_sampled_points, acting as Q
        tracks_for_certainty = tracks.permute(0, 2, 1, 3)  # [1, N, T, 2]
        track_logits_for_certainty = track_logits.permute(0, 2, 1, 3)  # [1, N, T, 512]
        raw_visible_logits_for_certainty = raw_visible_logits.permute(0, 2, 1, 3)  # [1, N, T, 1]

        # 4. Calculate certainty using the imported tracker_certainty function
        #    tracker_certainty typically returns [B, Q, T]
        pred_certainty = tracker_certainty(
            tracks_for_certainty,
            track_logits_for_certainty,
            certainty_radius  # Use the new parameter
        )  # Shape: [1, N, T] (batch, num_queries, num_frames)

        # 5. Combine raw visibility logits with certainty
        #    F.sigmoid(raw_visible_logits_for_certainty) is [1, N, T, 1]
        #    pred_certainty is [1, N, T, 1]
        combined_visibility = (
                                  F.sigmoid(raw_visible_logits_for_certainty) * pred_certainty
                              ) > certainty_threshold  # <--- 6. Use the new parameter
        # Shape: [1, N, T, 1] (boolean tensor)

        # 7. Prepare final outputs
        #    tracks_output: From [1, T, N, 2] to [T, N, 2]
        tracks = tracks.squeeze(0).float().detach().cpu().numpy()

        #    visibles_output: From [1, N, T, 1] boolean to [T, N, 1] float numpy
        #    Permute back to [1, T, N, 1] before squeezing the batch dimension
        visibles = combined_visibility.permute(0, 2, 1, 3).squeeze(0).float().detach().cpu().numpy()

        # Scale coordinates from resized_size back to original image size
        # Model output is (y, x) but we want (x, y) for opencv circles.
        tracks_scaled = np.zeros_like(tracks, dtype=np.float32)
        # tracks_scaled[..., 0] is x-coordinate
        tracks_scaled[..., 0] = tracks[..., 1] * (w_org / self.resized_size)  # x coord
        # tracks_scaled[..., 1] is y-coordinate
        tracks_scaled[..., 1] = tracks[..., 0] * (h_org / self.resized_size)  # y coord

        # Generate colors for visualization
        rand_color = viz_utils.get_colors(tracks.shape[1])  # N colors

        return tracks_scaled, visibles, rand_color


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample and fill 16-bit PNG depth maps using Warp.")
    parser.add_argument("--input_dir", type=str, default='realsense_example_glass',
                        help="Directory containing input RGB/mask files.")
    parser.add_argument("--num_points_per_frame", type=int, default=4,
                        help="Number of points to sample from each mask.")

    args = parser.parse_args()

    # --- Configuration ---
    INPUT_DIR = args.input_dir
    RGB_DIR = os.path.join(INPUT_DIR, 'rgb')
    MASK_DIR = os.path.join(INPUT_DIR, 'mask')
    OUT_DIR = os.path.join(INPUT_DIR, 'point_track_offline')  # Changed output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load all RGB and Mask file paths
    rgb_filepaths = sorted(glob.glob(os.path.join(RGB_DIR, "*.jpg")))[:64]  # Adjust pattern if needed
    mask_filepaths = sorted(glob.glob(os.path.join(MASK_DIR, "*.png")))[:64]

    if not rgb_filepaths:
        print(f"No RGB images found in {RGB_DIR}")
        exit()
    if not mask_filepaths:
        print(f"No mask images found in {MASK_DIR}")
        exit()

    # Load all RGB frames into a list
    all_rgb_frames = [np.array(Image.open(fp)) for fp in rgb_filepaths]

    # Load all mask frames into a list (or replicate the first if only one is provided)
    all_mask_frames = []
    if len(mask_filepaths) == 1:
        # If only one mask is provided, use it for all frames
        single_mask = np.array(Image.open(mask_filepaths[0])) / 255.0
        all_mask_frames = [single_mask] * len(all_rgb_frames)
    elif len(mask_filepaths) == len(rgb_filepaths):
        # If masks exist for all frames
        all_mask_frames = [np.array(Image.open(fp)) / 255.0 for fp in mask_filepaths]
    else:
        print("Error: Number of mask files does not match number of RGB files, and it's not a single mask file.")
        exit()

    point_tracker = PointTracker()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        # Perform offline tracking for the entire video
        all_tracks, all_visibles, rand_color = point_tracker.process_video(
            all_rgb_frames, all_mask_frames, num_points_per_frame=args.num_points_per_frame
        )

    # Visualize and save results for each frame
    print("Saving tracked frames...")
    for frame_idx in range(len(all_rgb_frames)):
        current_rgb_frame = all_rgb_frames[frame_idx].copy()  # Make a copy to draw on
        current_mask_frame = all_mask_frames[frame_idx]
        tracks_for_frame = all_tracks[frame_idx]  # (N, 2)
        visibles_for_frame = all_visibles[frame_idx]  # (N, 1)

        # Iterate through all tracked points for the current frame
        for point_idx in range(tracks_for_frame.shape[0]):
            if visibles_for_frame[point_idx] > 0.5:  # Check if the point is visible
                xy = tracks_for_frame[point_idx].astype(np.int32).tolist()
                if current_mask_frame[xy[1], xy[0]] > 0.5:
                    cv2.circle(current_rgb_frame, tuple(xy), 3, rand_color[point_idx], -1)

        output_filename = os.path.join(OUT_DIR, f"{frame_idx:06d}.png")
        cv2.imwrite(output_filename, current_rgb_frame[..., ::-1])  # Save as BGR (OpenCV default)

        # Optional: Display frame (uncomment if you have a display)
        # cv2.imshow("Tracked Frame", current_rgb_frame[..., ::-1])
        # cv2.waitKey(1)

    # cv2.destroyAllWindows()
    print(f"Tracking complete. Results saved to {OUT_DIR}")