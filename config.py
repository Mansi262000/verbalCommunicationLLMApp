# config.py

import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_community.llms import Ollama

def setup_models():
    """Downloads and initializes the required models."""
    print("Setting up models...")
    
    # Load Whisper model and processor
    global processor, whisper_model
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    
    # Load Ollama LLM model
    global llm

    
    llm = Ollama(model="mistral")
    
    print("Models loaded successfully.")

def check_gpu():
    """Checks if GPU is available and sets the appropriate device."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device

# Run setup when the script is executed
if __name__ == "__main__":
    setup_models()
    check_gpu()