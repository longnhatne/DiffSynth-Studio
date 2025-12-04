import csv
import os
from pathlib import Path
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import av
import numpy as np

def load_video_frames(video_path, max_frames=16):
    """Load frames from video file"""
    container = av.open(video_path)
    total_frames = container.streams.video[0].frames

    # Sample frames evenly
    indices = np.linspace(0, total_frames - 1, min(max_frames, total_frames), dtype=int)

    frames = []
    for i, frame in enumerate(container.decode(video=0)):
        if i in indices:
            frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) >= max_frames:
            break

    container.close()
    return frames

def caption_video(video_path, model, processor):
    """Generate caption for a video using Qwen2-VL"""
    try:
        print(f"Processing {video_path}...")

        # Prepare the conversation message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": video_path,
                        "max_pixels": 360 * 420,
                        "fps": 1.0,
                    },
                    {"type": "text", "text": "Describe this video in detail. Focus on the actions, characters, environment, lighting, and visual style. Provide a comprehensive description suitable for text-to-video generation."},
                ],
            }
        ]

        # Preparation for inference
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Generate caption
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text[0].strip()

    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return ""

def main():
    print("Loading Qwen2-VL model...")

    # Load model
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    print("Model loaded successfully!")

    # Read the CSV file
    csv_path = 'data/f5_multiVACE/metadata_vace.csv'
    video_dir = 'data/f5_multiVACE/videos'

    # Read existing CSV
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        rows = list(reader)

    # Add 'prompt' to fieldnames if not already there
    if 'prompt' not in fieldnames:
        fieldnames = list(fieldnames) + ['prompt']

    # Generate captions for each video
    for i, row in enumerate(rows):
        video_file = row['video']
        video_path = os.path.join(video_dir, os.path.basename(video_file))

        if os.path.exists(video_path):
            print(f"\n[{i+1}/{len(rows)}] Captioning {video_file}...")
            caption = caption_video(video_path, model, processor)
            row['prompt'] = caption
            print(f"Caption: {caption}")
        else:
            print(f"Video not found: {video_path}")
            row['prompt'] = ""

    # Write the updated CSV
    output_path = 'data/f5_multiVACE/metadata_vace.csv'
    with open(output_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n✓ Updated CSV saved to {output_path}")
    print(f"✓ Processed {len(rows)} videos")

if __name__ == "__main__":
    main()
