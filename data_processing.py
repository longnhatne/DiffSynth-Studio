import torch
from easy_dwpose import DWposeDetector
from PIL import Image
import requests
from controlnet_aux import LineartDetector, MidasDetector
from diffusers.utils import export_to_video
import cv2
import numpy as np
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# load checkpoints
lineart = LineartDetector.from_pretrained("lllyasviel/Annotators").to(device)
midas = MidasDetector.from_pretrained("lllyasviel/Annotators").to(device)
dwpose = DWposeDetector(device=device)


def process_frame(frame):
    '''
    frame: PIL.Image.Image
    '''
    skeleton_img = dwpose(
            frame,
            detect_resolution=frame.width,
            output_type="pil",
            include_hands=True,
            include_face=True,
        )

    depth_img = midas(frame, output_type="pil")
    lineart_img = lineart(frame, output_type="pil")

    # Invert lineart_img (PIL.Image)
    lineart_img = Image.fromarray(255 - np.array(lineart_img))
    
    if depth_img.size != frame.size:
        depth_img = depth_img.resize(frame.size, resample=Image.BILINEAR)
    if lineart_img.size != frame.size:
        lineart_img = lineart_img.resize(frame.size, resample=Image.BILINEAR)
    if skeleton_img.size != frame.size:
        skeleton_img = skeleton_img.resize(frame.size, resample=Image.BILINEAR)
    
    output = {
        "skeleton": skeleton_img,
        "depth": depth_img,
        "lineart": lineart_img,
    }   
    
    return output


def extract_character_mask(frame, return_alpha=False, blue_threshold=30, saturation_threshold=50):
    """
    Extract character mask from blue screen video frame using blue channel dominance.
    
    This approach detects pixels where the blue channel is significantly higher than
    both red and green channels, making it more robust to lighting variations.
    
    Args:
        frame: PIL.Image.Image - Input frame with blue screen
        return_alpha: bool - If True, returns RGBA image with alpha channel
                             If False, returns RGB image (white character on black)
        blue_threshold: int - Minimum difference between blue and other channels (default: 30)
        saturation_threshold: int - Minimum saturation for blue screen pixels (default: 50)
    
    Returns:
        PIL.Image.Image - Mask image
    """
    # Convert PIL to numpy array (RGB)
    frame_np = np.array(frame)
    
    # Split into RGB channels
    r, g, b = frame_np[:, :, 0], frame_np[:, :, 1], frame_np[:, :, 2]
    
    # 1. Blue channel dominance: blue is significantly higher than red AND green
    blue_dominant = (b.astype(np.float32) > r + blue_threshold) & \
                    (b.astype(np.float32) > g + blue_threshold)
    
    # 2. Additional saturation check to avoid gray areas
    # Convert to HSV to check saturation
    hsv = cv2.cvtColor(frame_np, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]
    high_saturation = saturation > saturation_threshold
    
    # Combine conditions: blue dominant AND high saturation = blue screen
    blue_screen_mask = (blue_dominant & high_saturation).astype(np.uint8) * 255
    
    # 3. Invert to get character (character = NOT blue screen)
    character_mask = cv2.bitwise_not(blue_screen_mask)
    
    # 4. Morphological cleanup to remove noise and smooth edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_CLOSE, kernel)
    character_mask = cv2.morphologyEx(character_mask, cv2.MORPH_OPEN, kernel)
    
    # 5. Optional: Additional refinement - remove small isolated regions
    # Find contours and filter by area
    contours, _ = cv2.findContours(character_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Find the largest contour (likely the main character)
        largest_contour = max(contours, key=cv2.contourArea)
        # Create a new mask with only contours above a certain area threshold
        min_area = frame_np.shape[0] * frame_np.shape[1] * 0.001  # 0.1% of image area
        refined_mask = np.zeros_like(character_mask)
        for contour in contours:
            if cv2.contourArea(contour) > min_area:
                cv2.drawContours(refined_mask, [contour], -1, 255, -1)
        character_mask = refined_mask
    
    if return_alpha:
        # Return RGBA image with alpha channel
        frame_rgba = cv2.cvtColor(frame_np, cv2.COLOR_RGB2RGBA)
        frame_rgba[:, :, 3] = character_mask  # Set alpha channel
        return Image.fromarray(frame_rgba, mode='RGBA')
    else:
        # Return RGB image: white character on black background
        mask_3ch = cv2.cvtColor(character_mask, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(mask_3ch, mode='RGB')


def get_video_metadata(video_path):
    """
    Get video metadata including FPS, resolution, and codec.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        dict: Video metadata with keys 'fps', 'width', 'height', 'fourcc', 'frame_count'
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    
    metadata = {
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'fourcc': int(cap.get(cv2.CAP_PROP_FOURCC)),
        'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    cap.release()
    return metadata


def read_video_as_pil_frames(video_path):
    """
    Reads a video from video_path and returns a list of PIL.Image.Image frames.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file {video_path}")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert from BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)
    cap.release()
    return frames


# Dataset root directory
dataset_root = "processed_f5_data_multiVACE"
videos_dir = os.path.join(dataset_root, "videos")
character_bluescreen_dir = os.path.join(dataset_root, "character_bluescreen")

# Create output directories if they don't exist
os.makedirs("data/f5_multiVACE/skeleton", exist_ok=True)
os.makedirs("data/f5_multiVACE/depth", exist_ok=True)
os.makedirs("data/f5_multiVACE/lineart", exist_ok=True)
os.makedirs("data/f5_multiVACE/masks", exist_ok=True)

# Get all video files from the videos directory
video_files = sorted([f for f in os.listdir(videos_dir) if f.endswith('.mp4')])
print(f"Found {len(video_files)} videos to process\n")

# Process each video
for video_file in video_files:
    video_path = os.path.join(videos_dir, video_file)

    # Extract video ID (first 5 digits) and base name
    video_basename = os.path.splitext(video_file)[0]
    video_id = video_basename.split('_')[0]  # e.g., '00050' from '00050_integration_XH'

    print(f"=" * 80)
    print(f"Processing main video: {video_basename} (ID: {video_id})")
    print(f"=" * 80)

    # Get original video metadata (FPS, resolution, etc.)
    try:
        video_metadata = get_video_metadata(video_path)
        print(f"  FPS: {video_metadata['fps']}")
        print(f"  Resolution: {video_metadata['width']}x{video_metadata['height']}")
        print(f"  Frame count: {video_metadata['frame_count']}")
    except Exception as e:
        print(f"  ✗ Error reading video metadata: {e}")
        continue

    # Read video frames
    try:
        video_frames = read_video_as_pil_frames(video_path)
    except Exception as e:
        print(f"  ✗ Error reading video frames: {e}")
        continue

    # Get original FPS
    original_fps = video_metadata['fps']

    # Define output paths
    skeleton_output = f"data/f5_multiVACE/skeleton/{video_basename}_skeleton.mp4"
    depth_output = f"data/f5_multiVACE/depth/{video_basename}_depth.mp4"
    lineart_output = f"data/f5_multiVACE/lineart/{video_basename}_lineart.mp4"

    # Process frames and collect outputs
    print(f"\nProcessing {len(video_frames)} frames...")
    skeleton_frames = []
    depth_frames = []
    lineart_frames = []

    for frame in video_frames:
        output = process_frame(frame)
        skeleton_frames.append(output["skeleton"])
        depth_frames.append(output["depth"])
        lineart_frames.append(output["lineart"])

    # Export videos using diffusers
    print("Exporting videos...")
    export_to_video(skeleton_frames, skeleton_output, fps=original_fps)
    export_to_video(depth_frames, depth_output, fps=original_fps)
    export_to_video(lineart_frames, lineart_output, fps=original_fps)

    print(f"✓ Saved: {skeleton_output}")
    print(f"✓ Saved: {depth_output}")
    print(f"✓ Saved: {lineart_output}")

    # Find and process corresponding character bluescreen videos
    character_video_dir = os.path.join(character_bluescreen_dir, video_id)

    if os.path.exists(character_video_dir):
        character_video_files = sorted([f for f in os.listdir(character_video_dir) if f.endswith('.mp4')])
        print(f"\nFound {len(character_video_files)} character videos for ID {video_id}")

        # Process each character bluescreen video
        for char_video_file in character_video_files:
            bluescreen_video_path = os.path.join(character_video_dir, char_video_file)

            # Extract character name from video path
            char_basename = os.path.splitext(char_video_file)[0]

            try:
                # Get metadata for each character video
                char_metadata = get_video_metadata(bluescreen_video_path)
                print(f"\n  Processing character video: {char_basename}")
                print(f"    FPS: {char_metadata['fps']}")
                print(f"    Resolution: {char_metadata['width']}x{char_metadata['height']}")

                bluescreen_video_frames = read_video_as_pil_frames(bluescreen_video_path)

                # Define output path
                mask_output = f"data/f5_multiVACE/masks/{char_basename}_mask.mp4"

                # Process frames and collect mask outputs
                print(f"    Processing {len(bluescreen_video_frames)} frames...")
                mask_frames = []
                for frame in bluescreen_video_frames:
                    mask_pil = extract_character_mask(frame)
                    mask_frames.append(mask_pil)

                # Export mask video using diffusers
                export_to_video(mask_frames, mask_output, fps=char_metadata['fps'])
                print(f"    ✓ Saved: {mask_output}")
            except Exception as e:
                print(f"    ✗ Error processing {char_basename}: {e}")
                continue
    else:
        print(f"\n  No character videos found for ID {video_id}")

    print()  # Add blank line between videos

print("\n" + "=" * 80)
print("Processing complete!")
print("=" * 80)