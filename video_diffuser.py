import cv2
import torch
import torchvision.transforms as transforms
from diffusers import DDPMPipeline
import numpy as np


# Load a video and preprocess it
def load_video(video_path, max_frames=16, frame_size=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(frame_size),
        transforms.ToTensor()
    ])

    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or count >= max_frames:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frame_tensor = transform(frame)
        frames.append(frame_tensor)
        count += 1

    cap.release()
    video_tensor = torch.stack(frames).unsqueeze(0)  # Shape: (1, num_frames, 3, H, W)
    return video_tensor


# Save generated frames back into a video
def save_video(frames, output_path, frame_size=(128, 128), fps=15):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

    for frame in frames:
        frame_np = frame.permute(1, 2, 0).cpu().numpy() * 255  # Convert to NumPy array
        frame_bgr = cv2.cvtColor(frame_np.astype('uint8'), cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()


# Use a pretrained diffusion model to generate frames
def video_diffusion(pipeline, video_tensor, num_inference_steps=50):
    num_frames, channels, height, width = video_tensor.shape[1:]
    generated_video = []

    for i in range(num_frames):
        frame = video_tensor[:, i, :, :, :].unsqueeze(0).to(pipeline.device)  # Move frame to the correct device
        generated_frame = pipeline(frame, num_inference_steps=num_inference_steps).images
        generated_video.append(generated_frame[0])

    return torch.stack(generated_video)


# Main execution flow
if __name__ == "__main__":
    # Load the pre-trained DDPM pipeline
    pipeline = DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")
    pipeline.to("cuda")  # Move the model to GPU if available

    # Load and preprocess the video into a tensor
    video_tensor = load_video('input/traffic.mp4')  # Replace with the path to your video
    video_tensor = video_tensor.to("cuda")  # Move the video tensor to GPU

    # Generate new video frames using the diffusion model
    generated_video = video_diffusion(pipeline, video_tensor)

    # Save the generated video frames to an output file
    save_video(generated_video, 'generated_video.mp4')
