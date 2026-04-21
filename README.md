# 🩺 MedBot — AI Medical Assistant

## 🚀 Overview
MedBot is a Streamlit-based AI chatbot that provides general health information using a Retrieval-Augmented Generation (RAG) pipeline and LLM integration.

---

## 🎯 Features

- AI-powered medical Q&A
- RAG-based document grounding
- Chat history tracking
- Clean SaaS-style UI
- PDF export of conversations
- FAQ-based quick queries

---

## 🧠 Tech Stack

- Frontend: Streamlit
- Backend: LangGraph, LangChain
- LLM: Groq (LLaMA3)
- Vector DB: FAISS
- PDF: ReportLab

---

## ⚙️ Architecture

User Query → Router → RAG Retrieval → LLM → Response → UI

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/medbot.git
cd medbot
pip install -r requirements.txt
