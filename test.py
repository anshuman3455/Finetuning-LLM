# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM

# # ----------------------------
# # Load tokenizer & model
# # ----------------------------
# model_path = "model/merged"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(
#     model_path,
#     device_map="auto",
#     torch_dtype=torch.float16
# )

# # Ensure padding token exists
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token

# # ----------------------------
# # Prompt in exact training format
# # ----------------------------
# prompt = """<|user|>
# Explain inflation in simple words.
# <|assistant|>
# """

# inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

# # ----------------------------
# # Generate output
# # ----------------------------
# outputs = model.generate(
#     **inputs,
#     max_new_tokens=150,
#     temperature=0.7,
#     top_p=0.9,
#     do_sample=True,
#     pad_token_id=tokenizer.eos_token_id
# )

# decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Remove prompt to get response only
# response = decoded.replace(prompt, "").strip()

# print("=== MODEL OUTPUT ===")
# print(response)


from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

MODEL_DIR = "model/merged"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

# Add special tokens if needed
special_tokens = ["<|user|>", "<|assistant|>"]
tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

# Load merged model
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True, device_map="auto")

# Resize embeddings if new tokens were added
model.resize_token_embeddings(len(tokenizer))

# Create pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True
)

# Test generation
res = pipe("What is inflation?", max_new_tokens=150)
print(res[0]["generated_text"])