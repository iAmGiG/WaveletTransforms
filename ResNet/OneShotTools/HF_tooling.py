# %%
import requests
import os

print(os.getenv("HUGGINGFACE_HUB_TOKEN"))

headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACE_HUB_TOKEN')}"}
response = requests.get(
    "https://huggingface.co/api/datasets/imagenet-1k", headers=headers)
print(response.status_code)
print(response.json())
