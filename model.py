from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
    # Removed 'device_map' to force model to load on CPU
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

model.eval()  # Set model to evaluation mode

# Take prompt from user input
prompt = input("Enter your prompt: ")

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate response
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=100)

# Decode the generated tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract the assistant's reply
assistant_response = generated_text.split("assistant", 1)[-1].strip()

print("Qwen:", assistant_response)
