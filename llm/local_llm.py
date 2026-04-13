
import requests
import time

class LocalLLM:
    def __init__(self):
        self.url = "http://127.0.0.1:8002/generate"

    def generate(self, user_input):
        start = time.time()

        # wrap user input in chat format
        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

        res = requests.post(self.url, json={"prompt": prompt})
        data = res.json()

        print("🔥 LLM RAW RESPONSE:", data)

        output = data.get("response", "")

        latency = time.time() - start
        return output, latency