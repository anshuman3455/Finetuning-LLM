import requests
import time

class LocalLLM:
    def __init__(self):
        self.url = "http://127.0.0.1:8002/generate"

    def generate(self, user_input):
        start = time.time()

        prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

        try:
            res = requests.post(self.url, json={"prompt": prompt}, timeout=30)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            print(" LLM ERROR:", e)
            return "LLM failed", 0

        print("LLM RAW RESPONSE:", data)

        output = data.get("response", "").strip()

        if not output:
            output = " Model returned empty response"

        latency = time.time() - start
        return output, latency