import torch
from diffusers import StableDiffusionPipeline

def main():
    # Load the pre-trained model
    model_id = "CompVis/stable-diffusion-v1-4"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = StableDiffusionPipeline.from_pretrained(model_id)
    pipe = pipe.to(device)

    # Prompt the user for input
    prompt = input("Enter the description for the image you want to generate: ")

    # Additional settings for improving quality
    num_inference_steps = int(input("Enter the number of inference steps (default 50): ") or "50")
    guidance_scale = float(input("Enter the guidance scale (default 7.5): ") or "7.5")

    # Generate the image
    print("Generating image... Please wait.")
    with torch.autocast("cuda"):
        image = pipe(prompt, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale).images[0]

    # Save the image
    image_filename = "generated_image.png"
    image.save(image_filename)
    print(f"Image saved as {image_filename}")

    # Display the image (optional)
    image.show()

if __name__ == "__main__":
    main()