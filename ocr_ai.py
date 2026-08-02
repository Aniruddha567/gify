import os
import requests
from huggingface_hub import InferenceClient

# ==================== CONFIGURATION ====================
OCR_API_KEY = "K83734259588957"
HF_API_TOKEN = "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"
IMAGE_PATH = "test_image.png"
# =======================================================

def perform_ocr(image_path, api_key):
    """Sends the image to OCR.space and returns the extracted text."""
    url = "https://api.ocr.space/parse/image"
    
    with open(image_path, 'rb') as image_file:
        payload = {
            'apikey': api_key,
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': '2'
        }
        files = {'file': image_file}
        
        print("🔄 Extracting text from image via OCR.space...")
        response = requests.post(url, data=payload, files=files)
        
        if response.status_code == 200:
            result = response.json()
            if result.get("IsErroredOnProcessing"):
                print("❌ OCR Error:", result.get("ErrorMessage"))
                return None
            
            parsed_results = result.get("ParsedResults", [])
            if parsed_results:
                return parsed_results[0].get("ParsedText")
        else:
            print(f"❌ OCR Connection Failed. Status code: {response.status_code}")
        return None


def analyze_chat_insights(chat_text, hf_token):
    """Sends the extracted chat text to a Hugging Face LLM for structured insights."""
    print("🧠 Querying Hugging Face AI for insights...")
    
    # Initialize the serverless inference client using Llama 3 (excellent for analysis)
    client = InferenceClient(
        model="meta-llama/Llama-3.1-8B-Instruct", 
        token=hf_token
    )
    
    # We craft a system prompt guiding the AI to extract exactly what we need
    system_prompt = (
        "You are an expert AI communication and psychology analyst. "
        "Analyze the provided raw OCR text from a chat screenshot and return a structured analysis. "
        "Focus on answering:\n"
        "1. Who are the participants in this conversation?\n"
        "2. For each person, what is their tone (e.g., casual, stressed, excited)?\n"
        "3. What are their primary intentions or goals in this chat?\n"
        "4. What is the emotional dynamic of the group overall?\n"
        "Keep the output professional, concise, and structured with clear markdown headings."
    )
    
    user_prompt = f"Here is the raw extracted chat text to analyze:\n\n{chat_text}"
    
    try:
        response = client.chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=800,
            temperature=0.7
        )
        # Print the AI's response
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ Failed to get AI insights: {str(e)}"


# --- Execution Flow ---
if __name__ == "__main__":
    # 1. Run OCR
    extracted_text = perform_ocr(IMAGE_PATH, OCR_API_KEY)
    
    if extracted_text:
        print("\n--- Raw OCR Text Output ---")
        print(extracted_text)
        print("----------------------------\n")
        
        # 2. Run AI Analysis
        analysis_report = analyze_chat_insights(extracted_text, HF_API_TOKEN)
        
        print("\n================ AI INSIGHTS REPORT ================")
        print(analysis_report)
        print("====================================================")
    else:
        print("Failed to extract any text from the image.")