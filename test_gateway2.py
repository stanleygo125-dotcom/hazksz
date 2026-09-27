from dotenv import load_dotenv
load_dotenv()
import requests, os

key = os.environ["LLM_GATEWAY_API_KEY"]
model = os.environ["LLM_MODEL"]
url = "https://api.softwaresystems.app/v1/chat/completions"
payload = {"model": model, "messages": [{"role": "user", "content": "hi"}]}

print("--- Attempt 1: Authorization: Bearer ---")
r1 = requests.post(url, headers={"Authorization": "Bearer " + key}, json=payload)
print(r1.status_code, r1.text)

print()
print("--- Attempt 2: X-API-Key ---")
r2 = requests.post(url, headers={"X-API-Key": key}, json=payload)
print(r2.status_code, r2.text)
