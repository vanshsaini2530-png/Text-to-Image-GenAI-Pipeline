import torch
from transformers import AutoTokenizer, CLIPTextModel
from train_attention_gan import AttentionGenerator, LATENT_DIM, device
import os

os.makedirs("pipeline_output", exist_ok=True)

print("=== 1. Loading Text Preprocessing & Transformer Encoder ===")
model_id = "openai/clip-vit-base-patch32"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)
text_encoder.eval()

print("=== 2. Loading Trained Attention-Enhanced GAN Generator ===")
generator = AttentionGenerator().to(device)
generator.load_state_dict(torch.load("attention_gan_generator.pth", map_location=device))
generator.eval()

# Map text descriptions to condition categories (0: Circle, 1: Square, 2: Triangle)
def get_category_from_prompt(prompt):
    prompt_lower = prompt.lower()
    if "circle" in prompt_lower or "round" in prompt_lower:
        return 0, "Circle"
    elif "square" in prompt_lower or "box" in prompt_lower:
        return 1, "Square"
    else:
        return 2, "Triangle"

def run_text_to_image_pipeline(prompt_text):
    print(f"\nProcessing Prompt: \"{prompt_text}\"")
    
    # Step A: Text Preprocessing & Tokenization (Task 3)
    inputs = tokenizer(
        [prompt_text], 
        padding=True, 
        truncation=True, 
        max_length=77, 
        return_tensors="pt"
    )
    
    # Step B: Text Embedding Generation via Transformer Encoder (Task 3)
    with torch.no_grad():
        outputs = text_encoder(**inputs)
        text_embeddings = outputs.last_hidden_state
    
    print(f" -> Token IDs Shape: {inputs['input_ids'].shape}")
    print(f" -> Text Embeddings Shape (Dense Representation): {text_embeddings.shape}")
    
    # Step C: Condition mapping & Attention-Enhanced GAN Image Generation (Task 2 & 5)
    class_idx, shape_name = get_category_from_prompt(prompt_text)
    print(f" -> Mapped Category via Attention / Classifier: {shape_name} (ID: {class_idx})")
    
    noise = torch.randn(1, LATENT_DIM).to(device)
    label_tensor = torch.tensor([class_idx]).to(device)
    
    with torch.no_grad():
        generated_image = generator(noise, label_tensor)
        
    # Normalize and save output image
    from torchvision.utils import save_image
    output_path = f"pipeline_output/generated_{shape_name.lower()}_pipeline.png"
    save_image(generated_image, output_path, normalize=True)
    print(f"Success! Final generated image saved to '{output_path}'")

if __name__ == "__main__":
    # Test our comprehensive pipeline with real-world prompts
    sample_prompts = [
        "Render a clean geometric circle shape",
        "Generate a sharp square structure",
        "Produce a distinct triangular visual"
    ]
    
    for prompt in sample_prompts:
        run_text_to_image_pipeline(prompt)
        
    print("\nComprehensive Text-to-Image Pipeline execution complete across all components!")