from flask import Flask, render_template, request
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)

# Load the model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.eval()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        prompt = request.form['prompt']
        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt")
        input_length = inputs['input_ids'].shape[1]
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=100)
        generated_tokens = outputs[0][input_length:]
        assistant_response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return render_template('index.html', prompt=prompt, response=assistant_response)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
