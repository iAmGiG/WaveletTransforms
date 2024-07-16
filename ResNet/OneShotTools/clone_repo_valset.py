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


def download_file_from_hf(url, token, destination):
    headers = {"Authorization": f"Bearer {token}"}
    try:
        response = requests.get(url, headers=headers, stream=True)
        # Raises stored HTTPError, if one occurred.
        response.raise_for_status()

        total_size_in_bytes = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        progress_bar = tqdm(total=total_size_in_bytes,
                            unit='iB', unit_scale=True)

        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()

        if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
            print("ERROR, something went wrong during the download")
        else:
            print(f"Downloaded {destination}")
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"An error occurred: {err}")


def main():
    token = get_hf_token()
    if token is None:
        print("Hugging Face token not found. Please ensure your token is stored correctly.")
        return

    val_url = "https://huggingface.co/datasets/ILSVRC/imagenet-1k/resolve/main/data/val_images.tar.gz?download=true"
    val_destination = "imagenet_data/val_images.tar.gz"

    if not os.path.exists(val_destination):
        os.makedirs(os.path.dirname(val_destination), exist_ok=True)
        download_file_from_hf(val_url, token, val_destination)
    else:
        print("Validation dataset already downloaded.")


if __name__ == "__main__":
    main()
