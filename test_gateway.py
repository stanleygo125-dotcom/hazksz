from dotenv import load_dotenv
load_dotenv()
import requests, os

r = requests.post(
    "https://api.softwaresystems.app/v1/chat/completions",
    headers={"Authorization": "Bearer " + os.environ["LLM_GATEWAY_API_KEY"]},
    json={"model": os.environ["LLM_MODEL"], "messages": [{"role": "user", "content": "hi"}]},
)
print(r.status_code)
print(r.text)
