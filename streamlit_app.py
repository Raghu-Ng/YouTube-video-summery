import streamlit as st
from pytube import YouTube
from transformers import pipeline
import torch
import os
from fpdf import FPDF

def download_audio(youtube_url):
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    audio_file = audio_stream.download(filename="audio.mp4")
    return audio_file

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def save_as_txt(summary_text):
    with open("summary.txt", "w", encoding="utf-8") as file:
        file.write(summary_text)
    return "summary.txt"

def save_as_pdf(summary_text):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output("summary.pdf")
    return "summary.pdf"

st.title("📹 YouTube Lecture Summarizer")
youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Summarize Video"):
    if youtube_url:
        st.info("Downloading audio...")
        audio_file = download_audio(youtube_url)
        
        st.info("Transcribing audio...")
        # Placeholder for transcription (use Whisper or other methods here)
        transcript = "Sample transcribed text from the lecture. Replace with actual transcription."
        
        st.info("Summarizing text...")
        summary = summarize_text(transcript)
        st.success("Summary Generated!")
        st.write(summary)
        
        # Download options
        txt_file = save_as_txt(summary)
        pdf_file = save_as_pdf(summary)
        
        st.download_button("Download Summary as .TXT", data=open(txt_file, "r").read(), file_name="summary.txt")
        st.download_button("Download Summary as .PDF", data=open(pdf_file, "rb").read(), file_name="summary.pdf", mime="application/pdf")
    else:
        st.warning("Please enter a valid YouTube URL.")
