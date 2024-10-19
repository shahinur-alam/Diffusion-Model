import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import matplotlib.pyplot as plt

def load_model():
    model_id = "CompVis/stable-diffusion-v1-4"
    try:
        if torch.cuda.is_available():
            pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            pipe = pipe.to("cuda")
            print("Model loaded on GPU.")
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_id)
            print("Model loaded on CPU. This may be slow.")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def generate_image(pipe, prompt, num_inference_steps=50):
    try:
        image = pipe(prompt, num_inference_steps=num_inference_steps).images[0]
        return image
    except Exception as e:
        print(f"Error generating image: {e}")
        return None

def display_image(image):
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def save_image(image, filename="generated_image.png"):
    try:
        image.save(filename)
        print(f"Image saved as {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

def main():
    # Load the model
    pipe = load_model()
    if pipe is None:
        return

    # Get user input
    prompt = input("Enter your text prompt: ")

    # Generate the image
    image = generate_image(pipe, prompt)
    if image is None:
        return

    # Display the image
    display_image(image)

    # Save the image
    save_image(image)

if __name__ == "__main__":
    main()