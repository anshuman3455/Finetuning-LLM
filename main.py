import os
import subprocess
import time
import json
from datetime import datetime

# CONFIG
MODEL_DIR = "model/merged"
GGUF_PATH = "model/model.gguf"
DATA_READY_FLAG = "data_ready.flag"   

# STEP 1: DATA PREP
def prepare_data():
    print("📦 Preparing dataset...")

    import os
    import json
    from datasets import load_dataset

    ds = load_dataset("Josephgflowers/Finance-Instruct-500k", split="train")
    ds = ds.select(range(5000))

    print("Sample keys:", ds[0].keys())

    formatted_data = []

    for x in ds:
        system = x.get("system", "")
        user = x.get("user", "")
        assistant = x.get("assistant", "")

        if not user.strip() or not assistant.strip():
            continue

        user_text = f"{system}\n{user}".strip()

        text = f"<|user|>\n{user_text}\n<|assistant|>\n{assistant}"

        formatted_data.append({"text": text})

    os.makedirs("data", exist_ok=True)

    file_path = "data/train.jsonl"

    with open(file_path, "w") as f:
        for row in formatted_data:
            f.write(json.dumps(row) + "\n")

    print(f"Dataset saved at {file_path}")
    print(f"Total samples: {len(formatted_data)}")


# STEP 2: TRAIN MODEL
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
def train_model():
    if os.path.exists(MODEL_DIR):
        print(" Model already trained")
        return

    print(" Training model...")

    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model
    from datasets import load_dataset
    import torch
    from transformers import BitsAndBytesConfig
    from transformers import AutoConfig



    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16,
    #     bnb_4bit_use_double_quant=True,
    #     bnb_4bit_quant_type="nf4",
    # )

    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)


    if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
        if "type" not in config.rope_scaling:
            config.rope_scaling["type"] = "linear"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # config=config,  
        device_map="mps",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    dataset = load_dataset(
        "json",
        data_files="data/train.jsonl"
    )

    def tokenize(example):
        texts = example["text"]

        input_ids = []
        attention_masks = []
        labels_list = []

        for text in texts:
            tokens = tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=512
            )

            labels = tokens["input_ids"].copy()

            # MASK USER PART
            if "<|assistant|>" in text:
                split_text = text.split("<|assistant|>")[0] + "<|assistant|>"

                prefix_ids = tokenizer(
                    split_text,
                    truncation=True,
                    max_length=512
                )["input_ids"]

                prefix_len = len(prefix_ids)

                labels[:prefix_len] = [-100] * prefix_len

            input_ids.append(tokens["input_ids"])
            attention_masks.append(tokens["attention_mask"])
            labels_list.append(labels)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": labels_list
        }

    dataset = dataset.map(tokenize, batched=True)

    train_dataset = dataset["train"]

    args = TrainingArguments(
        output_dir="model",
        per_device_train_batch_size=2,
        num_train_epochs=3,
        logging_steps=10
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset
    )
    trainer.train()

    model.save_pretrained("model/lora")

    print("Training complete")

# STEP 3.1: ADD SPECIAL TOKENS
def add_special_tokens():
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    
    special_tokens = ["<|user|>", "<|assistant|>"]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, trust_remote_code=True)

    model.resize_token_embeddings(len(tokenizer))

    tokenizer.save_pretrained(MODEL_DIR)
    model.save_pretrained(MODEL_DIR)

    print("Special tokens added and embeddings resized")

# STEP 3: MERGE MODEL
def merge_model():
    if os.path.exists(MODEL_DIR):
        print(" Model already merged")
        return

    print(" Merging LoRA...")

    from transformers import AutoModelForCausalLM
    from peft import PeftModel

    base = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        trust_remote_code=True)
    model = PeftModel.from_pretrained(base, "model/lora")

    model = model.merge_and_unload()

    model.save_pretrained(MODEL_DIR)

    print(" Merge complete")

# STEP 4: GGUF CONVERSION
def convert_to_gguf():
    if os.path.exists(GGUF_PATH):
        print(" GGUF already exists")
        return

    print("⚡ Converting to GGUF...")

    subprocess.run([
        "python",
        "llama.cpp/convert-hf-to-gguf.py",
        MODEL_DIR,
        "--outfile",
        GGUF_PATH
    ])

# STEP 5: START LLM SERVER

def start_llm_server():
    print(" Starting LLM server...")

    subprocess.Popen([
        "python", "-m", "uvicorn",
        "server:app",
        "--host", "127.0.0.1",
        "--port", "8002"
    ])

# STEP 6: START MCP SERVER

def start_mcp():
    print("Starting MCP server...")

    subprocess.Popen([
        "python",
        "-m",
        "uvicorn",
        "mcp.server:app",
        "--host", "127.0.0.1",
        "--port", "8000"
    ], cwd=os.getcwd())

# STEP 7: START UI

def start_ui():
    print(" Starting UI...")

    subprocess.Popen(["streamlit", "run", "ui/app.py"])

# STEP 8: EVALUATION
def evaluate():
    print("Running evaluation...")

    # simple evaluation
    from transformers import pipeline, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

    pipe = pipeline(
        "text-generation",
        model=MODEL_DIR,
        tokenizer=tokenizer,
        trust_remote_code=True
    )

    res = pipe("What is inflation?", max_new_tokens=50)

    print("Sample Output:", res[0]["generated_text"])

# MAIN PIPELINE
def main():
    start = time.time()

    prepare_data()
    train_model()
    merge_model()
    add_special_tokens()   

    evaluate()

    start_llm_server()
    start_mcp()

    time.sleep(5)

    start_ui()

    print(" FULL SYSTEM RUNNING")
    print(f"⏱ Total setup time: {time.time() - start:.2f} sec")

if __name__ == "__main__":
    main()