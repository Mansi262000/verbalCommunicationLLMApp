import streamlit as st
import torch
import torchaudio
import json
import sqlite3
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from langchain_community.llms import Ollama
from functools import lru_cache
import time
import numpy as np

# Load Whisper model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

# Initialize Ollama model
llm = Ollama(model="mistral")

# Database for tracking progress
conn = sqlite3.connect("progress.db")
cursor = conn.cursor()
cursor.execute("""CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    feedback TEXT,
    scores TEXT,
    tips TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)""")
conn.commit()

@lru_cache(maxsize=10)
def get_feedback(text):
    """Generates structured feedback report with scores and actionable tips."""
    prompt = f"""
    You are a communication expert. Evaluate the given text based on structure, delivery, and content.
    Provide a structured feedback report with:
    - Scores (out of 10) for clarity, tone, engagement
    - Actionable improvement tips
    
    Input: {text}
    
    Response:
    """
    response = llm.invoke(prompt)
    return response

def process_feedback(text):
    """Extracts structured feedback from LLM response."""
    feedback = get_feedback(text)
    scores = {"Clarity": 0, "Tone": 0, "Engagement": 0}
    tips = ""
    
    if "Scores:" in feedback:
        parts = feedback.split("Scores:")[1].strip().split("\n")
        for part in parts:
            if ":" in part:
                key, value = part.split(":")
                scores[key.strip()] = int(value.strip())
    
    if "Tips:" in feedback:
        tips = feedback.split("Tips:")[1].strip()
    
    return feedback, json.dumps(scores), tips

def impromptu_speaking(topic):
    """Generates a 2-minute impromptu speech."""
    prompt = f"Give a 2-minute impromptu speech on the topic: {topic}"
    return llm.invoke(prompt)

def storytelling(story):
    """Evaluates a story for narrative quality and engagement."""
    return process_feedback(story)

def conflict_resolution_scenario():
    """Simulates a conflict resolution scenario and evaluates responses."""
    prompt = "Your teammate is frustrated. How would you respond?"
    return llm.invoke(prompt)

def denoise_audio(audio_path):
    """Removes noise from audio using a basic low-pass filter."""
    waveform, sample_rate = torchaudio.load(audio_path)
    waveform = torchaudio.functional.lowpass_biquad(waveform, sample_rate, cutoff_freq=3000)
    torchaudio.save(audio_path, waveform, sample_rate)
    return audio_path

def transcribe_audio(audio_path):
    """Transcribes audio using Whisper."""
    denoised_audio = denoise_audio(audio_path)
    speech_array, _ = torchaudio.load(denoised_audio)
    input_features = processor(speech_array.squeeze().numpy(), return_tensors="pt").input_features
    
    retries = 3
    while retries > 0:
        try:
            with torch.no_grad():
                predicted_ids = whisper_model.generate(input_features)
            return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        except RuntimeError as e:
            retries -= 1
            if retries == 0:
                return "Error processing audio. Please try again."
            time.sleep(1)  # Wait before retrying

def voice_interface(audio):
    """Handles audio input: transcribes and provides structured feedback."""
    transcription = transcribe_audio(audio)
    feedback, scores, tips = process_feedback(transcription)
    return transcription, feedback, scores, tips

# Streamlit UI
st.title("AI-Powered Speech & Storytelling Feedback")

# Tabs for different functionalities
option = st.sidebar.selectbox("Choose a Feature", ["Text Feedback", "Audio Feedback", "Impromptu Speaking", "Storytelling", "Conflict Resolution"])

if option == "Text Feedback":
    text_input = st.text_area("Enter your text")
    if st.button("Get Feedback"):
        feedback, scores, tips = process_feedback(text_input)
        st.write("### Feedback:", feedback)
        st.write("### Scores:", scores)
        st.write("### Actionable Tips:", tips)

elif option == "Audio Feedback":
    audio_input = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if audio_input and st.button("Process Audio"):
        transcription, feedback, scores, tips = voice_interface(audio_input)
        st.write("### Transcription:", transcription)
        st.write("### Feedback:", feedback)
        st.write("### Scores:", scores)
        st.write("### Actionable Tips:", tips)

elif option == "Impromptu Speaking":
    topic = st.text_input("Enter a topic for impromptu speaking")
    if st.button("Generate Speech"):
        speech = impromptu_speaking(topic)
        st.write("### Generated Speech:", speech)

elif option == "Storytelling":
    story = st.text_area("Enter your story")
    if st.button("Evaluate Story"):
        feedback, scores, tips = storytelling(story)
        st.write("### Feedback:", feedback)
        st.write("### Scores:", scores)
        st.write("### Actionable Tips:", tips)

elif option == "Conflict Resolution":
    if st.button("Simulate Conflict Scenario"):
        scenario_response = conflict_resolution_scenario()
        st.write("### Scenario Response:", scenario_response)
