import base64
import json
import os
from pathlib import Path
import requests
from dotenv import load_dotenv

# Explicitly find and load the .env file in the current folder
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

hf_token = os.getenv("HF_TOKEN")
IMAGE_PATH = "test_image.png"  # Path to your WhatsApp screenshot
# =======================================================


def encode_image(image_path):
    """Encodes a local image to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def analyze_with_nuextract(image_path, token):
    if not token:
        return "❌ Error: HF_TOKEN not found in environment variables."

    url = "https://router.huggingface.co/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    base64_image = encode_image(image_path)
    data_url = f"data:image/png;base64,{base64_image}"

    # 1. Structural JSON template
    extraction_template = {
        "group_name": "verbatim-string",
        "chat_participants": ["string"],
        "messages": [
            {
                "sender": "string",  # 'You' or 'Adii'
                "message_text": "string",  # Hinglish message text
                "tone": "string",  # Tone: casual, dry, anxious, sarcastic, etc.
                "intention": "string",  # Core goal/intention behind the message
            }
        ],
        "overall_summary": "string",
    }

    # 2. System instructions + prompt framing
    system_prompt = (
        "You are an expert AI data extractor. Extract the WhatsApp chat details from the image "
        "and return strictly valid JSON strictly matching this template structure:\n"
        f"{json.dumps(extraction_template, indent=2)}\n\n"
        "Instructions:\n"
        "- Messages on the right side of the screen are sent by 'You' (the owner).\n"
        "- Parse Hinglish nuances accurately, especially context around 'CRT' (campus placement training)."
    )

    # 3. Build OpenAI-compatible payload
    payload = {
        "model": "numind/NuExtract3",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Extract all chat details from this screenshot using the JSON format provided.",
                    },
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": 0.4,
        "max_tokens": 1500,
    }

    print("🔄 Sending screenshot to NuExtract3 via Router...")
    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        try:
            result = response.json()
            output_content = result["choices"][0]["message"]["content"]
            return output_content
        except Exception as e:
            return f"Error parsing response: {e}\nRaw Response: {response.text}"
    else:
        return f"❌ Error {response.status_code}: {response.text}"

if __name__ == "__main__":
    extracted_data = analyze_with_nuextract(IMAGE_PATH, hf_token)
    print("\n================ NUEXTRACT3 STRUCTURED OUTPUT ================")
    print(extracted_data)
    print("===============================================================")