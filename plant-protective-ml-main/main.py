from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import google.generativeai as genai
import json
from dotenv import load_dotenv
import os
load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def detailed_prompt():
    return (
        "তুমি একজন অভিজ্ঞ কৃষি বিশেষজ্ঞ। আমি একটি গাছের পাতা অথবা ফসলের ছবি দেবো। "
        "ছবিটি দেখে নিচের তথ্যগুলো বাংলা ভাষায় স্পষ্ট ও প্রাঞ্জলভাবে JSON আকারে প্রদান করো:\n\n"
        "1. title: রোগের নাম বা সমস্যার শিরোনাম\n"
        "2. explanation: ছবির উপর ভিত্তি করে সমস্যা বিশ্লেষণ করো\n"
        "3. organic_solution: প্রাকৃতিক বা জৈব সমাধানসমূহ (list আকারে)\n"
        "4. chemical_solution: রাসায়নিক সমাধানসমূহ (list আকারে)\n"
        "5. preventive_measures: ভবিষ্যতে সমস্যা প্রতিরোধে করণীয় (list আকারে)\n"
        "6. crop_care_tips: সাধারণ ফসল পরিচর্যার টিপস (list আকারে)\n\n"
        "উত্তরটি শুধুমাত্র একটি সুন্দর ফরম্যাট করা JSON string হিসেবে দাও, যেনো Python কোডে তা সরাসরি ব্যবহার করা যায়।"
    )

def generate_flash_response(image_data):
    model = genai.GenerativeModel(
        "gemini-1.5-flash",
        generation_config={
            "max_output_tokens": 1500,
            "temperature": 0.4,
            "top_p": 0.85,
            "top_k": 50,
        }
    )

    response = model.generate_content([
        {"mime_type": "image/jpeg", "data": image_data},
        detailed_prompt()
    ])

    text = response.text.strip()

    if not text:
        raise ValueError("Gemini did not return any response.")

    start_idx = text.find("{")
    end_idx = text.rfind("}") + 1

    if start_idx == -1 or end_idx == -1:
        raise ValueError("Response is not in JSON format:\n" + text)

    json_text = text[start_idx:end_idx]
    return json.loads(json_text)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        response = generate_flash_response(image_data)
        return response
    except Exception as e:
        return {"error": f"Error: {str(e)}"}
