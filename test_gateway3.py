from dotenv import load_dotenv
load_dotenv()
import requests, os

key = os.environ["LLM_GATEWAY_API_KEY"]
model = os.environ["LLM_MODEL"]
url = "https://api.softwaresystems.app/api/chat"
payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "stream": False}

print("--- Attempt 1: Bearer, native /api/chat ---")
r1 = requests.post(url, headers={"Authorization": "Bearer " + key}, json=payload)
print(r1.status_code, r1.text[:500])

print()
print("--- Attempt 2: X-API-Key, native /api/chat ---")
r2 = requests.post(url, headers={"X-API-Key": key}, json=payload)
print(r2.status_code, r2.text[:500])
