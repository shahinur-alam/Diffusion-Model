import torch
import numpy as np
import imageio
from diffusers import DiffusionPipeline, DDIMScheduler
from video_diffusion_pytorch import Unet3D, GaussianDiffusion

# Check for GPU availability and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Part 1: Loading a Pretrained Model
def load_pretrained_model():
    pipeline = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipeline.to(device)
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_slicing()
    return pipeline


# Part 2: Generating Video
def generate_video(pipeline, prompt, output_file, height=320, width=576, num_frames=100):
    video_frames = pipeline(
        prompt,
        num_inference_steps=50,
        height=height,
        width=width,
        num_frames=num_frames
    ).frames

    print(f"Type of video_frames: {type(video_frames)}")
    print(f"Shape of video_frames: {video_frames.shape if hasattr(video_frames, 'shape') else 'N/A'}")
    print(f"Data type of video_frames: {video_frames.dtype if hasattr(video_frames, 'dtype') else 'N/A'}")

    # Convert frames to the correct format
    if isinstance(video_frames, torch.Tensor):
        video_frames = video_frames.cpu().numpy()
    if isinstance(video_frames, list):
        video_frames = np.array(video_frames)

    # Ensure frames are in the correct shape (T, H, W, C)
    if video_frames.ndim == 5:
        video_frames = video_frames.squeeze(0)  # Remove batch dimension if present

    # Normalize pixel values if necessary
    if video_frames.dtype == np.float32:
        video_frames = (video_frames * 255).astype(np.uint8)

    # Ensure we're in RGB format
    if video_frames.shape[-1] == 4:  # RGBA
        video_frames = video_frames[..., :3]  # Convert to RGB

    # Use imageio to write the video
    imageio.mimsave(output_file, video_frames, fps=10)
    print(f"Video generated and saved as '{output_file}'")


# Part 3: Fine-tuning for Custom Tasks
def setup_fine_tuning():
    model = Unet3D(
        dim=64,
        dim_mults=(1, 2, 4, 8)
    ).to(device)
    diffusion = GaussianDiffusion(
        model,
        image_size=64,  # frame size
        num_frames=100,  # number of frames
        timesteps=1000
    ).to(device)
    return diffusion


def train_step(diffusion, videos):
    videos = videos.to(device)
    loss = diffusion(videos)
    loss.backward()
    # Here you would typically update model parameters using an optimizer
    return loss.item()


# Part 4: Improving Temporal Consistency
def setup_temporal_consistency(pipeline):
    scheduler = DDIMScheduler.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        subfolder="scheduler",
        clip_sample=False,
        timestep_spacing="linspace",
        beta_schedule="linear",
        steps_offset=1,
    )
    pipeline.scheduler = scheduler
    return pipeline


# Main execution
if __name__ == "__main__":
    # Load pretrained model
    print("Loading pretrained model...")
    pipeline = load_pretrained_model()

    # Generate video with increased resolution
    print("Generating high-resolution video...")
    prompt = "A cat dancing with a song"
    generate_video(pipeline, prompt, "generated_high_res_video.mp4", height=480, width=640)

    # Setup for fine-tuning
    print("Setting up fine-tuning...")
    diffusion = setup_fine_tuning()

    # Simulated fine-tuning (you would replace this with your actual dataset)
    print("Simulating fine-tuning...")
    for i in range(5):  # Simulate 5 training steps
        videos = torch.randn(4, 3, 16, 64, 64).to(device)  # Simulated video data
        loss = train_step(diffusion, videos)
        print(f"Training step {i + 1}, Loss: {loss}")

    # Setup for improved temporal consistency
    print("Setting up model with improved temporal consistency...")
    temporal_pipeline = setup_temporal_consistency(pipeline)

    # Generate video with improved temporal consistency and high resolution
    print("Generating high-resolution video with improved temporal consistency...")
    generate_video(temporal_pipeline, prompt, "temporal_consistent_high_res_video.mp4", height=576, width=1024)

    print("Process completed!")