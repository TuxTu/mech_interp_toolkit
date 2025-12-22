from witt import load_model, load_tokenizer, tokenize

model_id = "Qwen/Qwen3-0.6B"

model = load_model(model_id)
tokenizer = load_tokenizer(model_id)

messages = [
    {"role": "system", "content": "You are Qwen, a helpful AI."},
    {"role": "user", "content": "Explain quantum physics in one sentence."}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

print(text)