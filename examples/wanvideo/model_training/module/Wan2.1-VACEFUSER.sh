CUDA_VISIBLE_DEVICES=0 accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/f5_multiVACE/ \
  --dataset_metadata_path data/f5_multiVACE/metadata_vace.csv \
  --data_file_keys "video,vace_video,vace_video_mask,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 20 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 10 \
  --remove_prefix_in_ckpt "pipe.vace_fuser." \
  --output_path "./models/train/Wan2.1-VACE-1.3B-FUSER" \
  --trainable_models "vace_fuser" \
  --extra_inputs "vace_video,vace_video_mask,vace_reference_image" \
#   --use_gradient_checkpointing_offload