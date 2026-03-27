"""Quick sanity check: send 1 image to each model, see if it responds."""
import os, requests, json
from dotenv import load_dotenv
load_dotenv()

key = os.environ["OPENROUTER_API_KEY"]
headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

models = ["google/gemma-3-27b-it"]

for model in models:
    try:
        r = requests.post("https://openrouter.ai/api/v1/chat/completions",
            headers=headers, json={
                "model": model,
                "messages": [{"role": "user", "content": [
                    {"type": "text", "text": "What do you see? Reply in 5 words."},
                    {"type": "image_url", "image_url": {"url": "https://i.redd.it/m16dhaqyply21.jpg"}}
                ]}],
                "max_tokens": 20
            }, timeout=30)
        data = r.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "EMPTY")
        print(f"OK  {model}: {content}")
    except Exception as e:
        print(f"ERR {model}: {e}")
