from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("model/merged")
model = AutoModelForCausalLM.from_pretrained(
    "model/merged",
    device_map="auto",
    torch_dtype=torch.float16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

class Request(BaseModel):
    prompt: str

@app.post("/generate")
async def generate(req: Request):
    prompt = req.prompt.strip()

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.8,   
        top_p=0.95,
        do_sample=True,
        repetition_penalty=1.1,  
        eos_token_id=None,       
        pad_token_id=tokenizer.eos_token_id
    )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if "<|assistant|>" in decoded:
        response = decoded.split("<|assistant|>")[-1]
    else:
        response = decoded.replace(prompt, "")

    response = response.replace("</s>", "").strip()

    if not response:
        response = " Model is not trained well yet. Try retraining."

    print("FINAL OUTPUT:", response)

    return {"response": response}