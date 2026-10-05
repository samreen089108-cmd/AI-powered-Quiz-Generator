# AI-Powered Quiz Generator from Notes

A Streamlit app that turns your study notes into quizzes and flashcards using a local LLM (Ollama) and RAG (Retrieval-Augmented Generation). Everything runs on your own laptop, so your notes never leave your machine.

## Features
- Upload notes as PDF, Word (.docx), TXT or MD, or paste text directly
- RAG pipeline: notes are chunked, embedded and searched, so questions come from your notes
- Flashcards: question on the front, answer and explanation on the back, with Next/Previous buttons
- Graded multiple-choice quiz with a score and explanations
- Chat with your notes, with the source passages shown
- Choose the number of questions, difficulty and an optional focus topic
- Download the quiz (with or without answers) as PDF, Word, HTML, TXT or JSON

## Tech Stack
Python, Streamlit, Ollama (llama3.2 and nomic-embed-text), NumPy, pypdf, python-docx, fpdf2, HTML/CSS

## How to Run
1. Install Python 3.12 or newer and Ollama (https://ollama.com/download)
2. Download the models:
```
   ollama pull llama3.2
   ollama pull nomic-embed-text
```
3. Install the libraries:
```
   pip install -r requirements.txt
```
4. Start the app:
```
   streamlit run app.py
```

## How It Works
1. Notes are split into overlapping chunks.
2. Each chunk is converted to a vector with `nomic-embed-text`.
3. The most relevant chunks are retrieved for a topic or a chat question.
4. `llama3.2` generates quiz questions or answers using only those chunks.
