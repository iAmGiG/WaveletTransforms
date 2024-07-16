import os
import requests
from tqdm import tqdm


def get_hf_token():
    """Read the Hugging Face token from the cache directory."""
    token_path = os.path.expanduser("~/.cache/huggingface/token")
    if os.path.exists(token_path):
        with open(token_path, "r") as file:
            return file.read().strip()
    return None


def download_large_file(url, token, destination):
    """Download a large file in chunks using requests with resume support."""
    headers = {'Authorization': f'Bearer {token}'}
    if os.path.exists(destination):
        # Get the file size of the partially downloaded file
        resume_header = {'Range': f'bytes={os.path.getsize(destination)}-'}
        headers.update(resume_header)

    response = requests.get(url, headers=headers, stream=True)
    total_size = int(response.headers.get('content-length', 0)
                     ) + os.path.getsize(destination)

    mode = 'ab' if 'bytes' in headers.get('Range', '') else 'wb'
    with open(destination, mode) as file, tqdm(
        desc=destination,
        total=total_size,
        initial=os.path.getsize(destination),
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))
    print(f'File downloaded successfully to {destination}')


# Set your Hugging Face token from the cache directory
hf_token = get_hf_token()

if hf_token:
    # Download the file using the Hugging Face Hub API
    try:
        file_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/test_images.tar.gz?download=true"
        save_path = "./test_images.tar.gz"
        print(f"Downloading file from URL: {file_url} with token: {hf_token}")
        download_large_file(file_url, hf_token, save_path)
    except Exception as e:
        print(f"Error occurred while downloading the file: {e}")
else:
    print("Hugging Face token not found. Please login using huggingface-cli.")
