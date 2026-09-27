import os
k = os.environ.get("LLM_GATEWAY_API_KEY", "")
print("length:", len(k))
print("starts:", repr(k[:6]))
print("ends:", repr(k[-6:]))
