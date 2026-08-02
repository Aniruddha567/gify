import requests

def perform_ocr(image_path, api_key):
    url = "https://api.ocr.space/parse/image"
    
    with open(image_path, 'rb') as image_file:
        payload = {
            'apikey': api_key,
            'language': 'eng',
            'isOverlayRequired': False,
            'OCREngine': '2'
        }
        
        files = {
            'file': image_file
        }
        
        print("Sending image to OCR.space...")
        response = requests.post(url, data=payload, files=files)
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("IsErroredOnProcessing"):
                print("Error:", result.get("ErrorMessage"))
                return
            
            parsed_results = result.get("ParsedResults", [])
            for page in parsed_results:
                print("\n--- Extracted Text ---")
                print(page.get("ParsedText"))
                print("----------------------")
        else:
            print(f"Failed to connect. Status code: {response.status_code}")
            print(response.text)


API_KEY = "K83734259588957" 

IMAGE_PATH = "test_image.png" 

perform_ocr(IMAGE_PATH, API_KEY)