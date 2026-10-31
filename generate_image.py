import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel

print("Loading base model...")
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float32)

print("Loading custom LoRA weights...")
# Load PEFT adapter weights seamlessly
pipe.unet = PeftModel.from_pretrained(pipe.unet, r"C:\Users\vansh\OneDrive\Desktop\lora-custom-model")

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe.to(device)

prompt = "a domain-specific solid red abstract background"
print(f"Generating image for prompt: '{prompt}'...")
image = pipe(prompt).images[0]

image.save("generated_custom_output.png")
print("Image generated and saved successfully as 'generated_custom_output.png'!")
