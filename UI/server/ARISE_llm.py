import re
import time
from fastapi import APIRouter
from pydantic import BaseModel
from llama_cpp import Llama
from fastapi.middleware.cors import CORSMiddleware


router = APIRouter()


# Initialize LLM
llm = Llama(
    model_path="../../models/SmolLM2-1.7B-Instruct-Q4_K_M.gguf",
    n_ctx=256,
    verbose=False,
    n_threads=4,
    n_batch=64
)

def generate_reply(prompt):
    llm_start_time = time.time()
    output = llm(
        f"<|system|>\nYou are a helpful voice assistant named arise. Respond clearly, naturally, and concisely as if spoken aloud. No markdown or formatting.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>",
        max_tokens=250,
        temperature=0.4
    )
    llm_end_time = time.time()
    print(f"LLM processing time: {llm_end_time - llm_start_time:.4f} seconds")
    return output['choices'][0]['text'].strip()

def clean_text_for_tts(text):
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'__(.*?)__', r'\1', text)
    text = re.sub(r'_(.*?)_', r'\1', text)
    text = re.sub(r'`{1,3}.*?`{1,3}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'/\w+\b', '', text)
    text = re.sub(r'[\[\]\{\}\(\)\|\\/]', '', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 to \2', text)
    text = re.sub(r'-', ' ', text)
    return text.strip()

class PromptRequest(BaseModel):
    prompt: str

@router.post("/reply")
async def llm_reply(request: PromptRequest):
    answer = generate_reply(request.prompt)
    # Optionally clean for TTS
    cleaned = clean_text_for_tts(answer)
    return {
        "response": answer,
        "cleaned_for_tts": cleaned
    }