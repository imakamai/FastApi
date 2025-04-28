import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# commandline_args = os.environ.get('COMMANDLINE_ARGS', "--skip-torch-cuda-test --no-half")

# Download the model and tokenizer from Hugging Face
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2",
                                             torch_dtype=torch.float32,
                                             trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2",
                                          trust_remote_code=True)

prompt = "What is the capital of the South Korea?"
inputs = tokenizer(prompt,
                   return_tensors="pt",
                   return_attention_mask=False,
                   add_special_tokens=False)
outputs = model.generate(**inputs,
                         max_length=32,
                         pad_token_id=tokenizer.eos_token_id)
text = tokenizer.batch_decode(outputs)[0]
print(text)