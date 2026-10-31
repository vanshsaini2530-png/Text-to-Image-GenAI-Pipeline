import torch
from transformers import AutoTokenizer, CLIPTextModel

print("Loading CLIP tokenizer and text encoder model from Hugging Face...")
model_id = "openai/clip-vit-base-patch32"

# Use the text-only model class to prevent image encoder errors
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_encoder = CLIPTextModel.from_pretrained(model_id)

# Sample text descriptions for our image generation pipeline
texts = [
    "A domain-specific solid red abstract background",
    "A sharp geometric square shape rendered cleanly"
]

print("\n--- 1. Tokenization Process ---")
# Tokenize text inputs
inputs = tokenizer(
    texts, 
    padding=True, 
    truncation=True, 
    max_length=77, 
    return_tensors="pt"
)

print(f"Input IDs shape (Token indices): {inputs['input_ids'].shape}")
print(f"Input IDs tokens for first text:\n{inputs['input_ids'][0]}")

print("\n--- 2. Encoding to Dense Embeddings ---")
# Pass tokenized representations through the text encoder model
with torch.no_grad():
    outputs = text_encoder(**inputs)
    text_embeddings = outputs.last_hidden_state

print(f"Final Text Embeddings Output Shape: {text_embeddings.shape}")
print("(Batch Size, Sequence Length, Hidden Dimension)")
print("\nText preprocessing and encoding pipeline completed successfully!")
