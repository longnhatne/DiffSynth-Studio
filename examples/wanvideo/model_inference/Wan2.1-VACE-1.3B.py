import torch
from PIL import Image
from diffsynth import save_video, VideoData
from diffsynth.pipelines.wan_video_new import WanVideoPipeline, ModelConfig
from modelscope import dataset_snapshot_download
from diffsynth.models.wan_video_vace import VaceFuser
from safetensors.torch import load_file

vace_fuser_path = "models/train/Wan2.1-VACE-1.3B-FUSER/epoch-3.safetensors"

pipe = WanVideoPipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="diffusion_pytorch_model*.safetensors", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="models_t5_umt5-xxl-enc-bf16.pth", offload_device="cpu"),
        ModelConfig(model_id="Wan-AI/Wan2.1-VACE-1.3B", origin_file_pattern="Wan2.1_VAE.pth", offload_device="cpu"),
    ],
)

pipe.vace_fuser = VaceFuser(
                num_vace_blocks=len(pipe.vace.vace_layers),
                dim=pipe.dit.dim,
                num_heads=pipe.dit.num_heads,
            ).to(torch.bfloat16).requires_grad_(False)

state_dict = load_file(vace_fuser_path)
state_dict = {k.replace("pipe.vace_fuser.", ""): v for k, v in state_dict.items()}
pipe.vace_fuser.load_state_dict(state_dict)


pipe.enable_vram_management()

# # Depth video -> Video
control_video = VideoData("data/f5_multiVACE/depth/00149_depth.mp4", height=480, width=832)
num_frames = len(control_video)

char1_mask = VideoData("data/f5_multiVACE/masks/00149_mathilda_mask.mp4", height=480, width=832)
char2_mask = VideoData("data/f5_multiVACE/masks/00149_lambert_mask.mp4", height=480, width=832)

char1_reference_image = Image.open("data/f5_multiVACE/character_images/mathilda/6.png").resize((832, 480))
char2_reference_image = Image.open("data/f5_multiVACE/character_images/lambert/1.png").resize((832, 480))

control_video = [control_video]
char_masks = [char1_mask, char2_mask]
char_reference_images = [char1_reference_image, char2_reference_image]
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_video=control_video,
#     seed=1, tiled=True
# )
# save_video(video, "video1.mp4", fps=15, quality=5)

# # Reference image -> Video
# video = pipe(
#     prompt="两只可爱的橘猫戴上拳击手套，站在一个拳击台上搏斗。",
#     negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
#     vace_reference_image=Image.open("data/examples/wan/cat_fightning.jpg").resize((832, 480)),
#     seed=1, tiled=True
# )
# save_video(video, "video2.mp4", fps=15, quality=5)

# Depth video + Reference image -> Video
video = pipe(
    prompt="",
    negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
    vace_video=control_video,
    vace_reference_image=char_reference_images,
    vace_video_mask=char_masks,
    seed=1, tiled=True, num_frames=num_frames
)
save_video(video, "video_multiVACE.mp4", fps=15, quality=5)
