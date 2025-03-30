import streamlit as st
import yt_dlp
from transformers import pipeline
import torch
import whisper
from fpdf import FPDF
import os

def download_audio(youtube_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'audio.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info_dict).replace(".webm", ".mp3").replace(".m4a", ".mp3")
        return filename
    except Exception as e:
        st.error(f"Error downloading audio: {e}")
        return None

def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0 if torch.cuda.is_available() else -1)
    summary = summarizer(text, max_length=150, min_length=50, do_sample=False)
    return summary[0]['summary_text']

def save_as_txt(summary_text):
    txt_file = "summary.txt"
    with open(txt_file, "w", encoding="utf-8") as file:
        file.write(summary_text)
    return txt_file

def save_as_pdf(summary_text):
    pdf_file = "summary.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.multi_cell(0, 10, summary_text)
    pdf.output(pdf_file)
    return pdf_file

st.title("📹 YouTube Lecture Summarizer")
youtube_url = st.text_input("Enter YouTube Video URL:")

if st.button("Summarize Video"):
    if youtube_url:
        st.info("Downloading audio...")
        audio_file = download_audio(youtube_url)
        
        if audio_file and os.path.exists(audio_file):
            st.info("Transcribing audio...")
            transcript = transcribe_audio(audio_file)
            
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
            st.error("Failed to download or process audio.")
    else:
        st.warning("Please enter a valid YouTube URL.")
