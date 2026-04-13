# server.py
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# ----------------------------
# Load tokenizer and model
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained("model/merged")
model = AutoModelForCausalLM.from_pretrained(
    "model/merged",
    device_map="auto",
    torch_dtype=torch.float16
)

# Ensure padding token exists
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class Request(BaseModel):
    prompt: str

# @app.post("/generate")
# async def generate(req: Request):
#     prompt = req.prompt.strip()
    
#     # Tokenize prompt
#     inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

#     # Generate output
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=150,
#         temperature=0.7,
#         top_p=0.9,
#         do_sample=True,
#         pad_token_id=tokenizer.eos_token_id,  # Important!
#     )

#     decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # Fallback: if slicing removes all text, return full decoded text
#     response = decoded[len(prompt):].strip()
#     if not response:
#         response = decoded.strip()

#     return {"response": response}


@app.post("/generate")
async def generate(req: Request):
    prompt = req.prompt.strip()

    # ✅ FORCE proper chat format
    if "<|assistant|>" not in prompt:
        prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id  # 🔥 IMPORTANT
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ✅ SAFE RESPONSE EXTRACTION
    if "<|assistant|>" in decoded:
        response = decoded.split("<|assistant|>")[-1].strip()
    else:
        response = decoded[len(prompt):].strip()

    print("✅ FINAL OUTPUT:", response)

    return {"response": response}