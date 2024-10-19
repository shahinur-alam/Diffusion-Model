from diffusers import StableDiffusionPipeline
import torch

# Load the model and move it to GPU if available
model_id = "CompVis/stable-diffusion-v1-4"
pipe = StableDiffusionPipeline.from_pretrained(model_id).to("cuda" if torch.cuda.is_available() else "cpu")

# Define the prompt (text input) for image generation
prompt = "Generate a funny, fluffy, fat tuxedo cat image sitting in a lovely, calm, and quiet place."

# Generate the image
image = pipe(prompt).images[0]

# Save the generated image
image.save("generated_image.png")
