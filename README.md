# 🩺 MedBot — Agentic AI Medical Assistant

MedBot is an agentic AI system designed to answer medical queries using a structured LangGraph pipeline. It combines Retrieval-Augmented Generation (RAG), intelligent routing, memory, and self-evaluation to provide accurate and context-aware responses.

---

## 🚀 Overview

MedBot addresses a critical problem: **users often receive unverified or hallucinated medical information from generic AI systems**.

This system improves reliability by:
- Grounding answers in a curated medical knowledge base
- Routing queries intelligently between retrieval and tools
- Maintaining conversational context
- Evaluating its own responses for faithfulness

Unlike a basic chatbot, MedBot behaves as an **agentic system**, making decisions about *how* to answer rather than just generating text.

---

## 🧠 System Architecture (LangGraph)

The system is built as a multi-step pipeline:

- **Memory Node**  
  Maintains conversation state using `thread_id` and `MemorySaver`.

- **Router Node**  
  Determines whether a query requires:
  - Retrieval (knowledge base)
  - Tool usage (real-time info)

- **Retrieval Node**  
  Fetches relevant context from ChromaDB.

- **Tool Node**  
  Handles real-time queries (e.g., latest updates).

- **Answer Node**  
  Generates final grounded responses using retrieved context.

- **Eval Node**  
  Computes faithfulness scores for response validation.

- **Save Node**  
  Stores conversation state for continuity.

---

## 📚 Knowledge Base

- **Total Documents:** 15  
- **Coverage Includes:**
  - Chronic diseases (Hypertension, Diabetes)
  - Respiratory conditions (Asthma, COPD)
  - Preventive care (Vaccines, Nutrition, Exercise)
  - Emergency awareness (Stroke, First Aid)
  - Public health topics (Antibiotic resistance, Mental health)

### Why this works:
- Focused, high-quality documents reduce hallucination
- Domain-specific coverage improves retrieval precision

---

## 🛠 Tool Integration

- Used for **real-time queries** (e.g., latest COVID updates)
- Triggered dynamically by the Router Node
- Ensures system remains **current beyond static knowledge base**

---

## 📊 Evaluation & Testing

### ✅ Test Results
- **Total Tests Passed:** 9 / 10  
- **Red-Team Tests Passed:** 2 / 2  
- **Average Faithfulness:** 0.84  

### 🔍 Example Success
- Stroke FAST signs → Highly accurate (0.91 faithfulness)

### ⚠️ Limitation
- Out-of-domain queries (e.g., crypto) still routed to retrieval

---

## 📈 RAGAS Metrics

| Metric             | Score |
|--------------------|-------|
| Faithfulness       | 0.84  |
| Answer Relevance   | 0.87  |
| Context Precision  | 0.80  |

### Interpretation:
- Strong grounding to source data
- High relevance to user queries
- Moderate retrieval precision (scope for improvement)

---

## 💻 Deployment

- Built with **Streamlit UI**
- Features:
  - Interactive chat interface
  - Conversational memory
  - Real-time response generation

---

## ⚠️ Limitations

- No hybrid retrieval (semantic only)
- Limited to curated knowledge base
- Routing can mis-handle unrelated queries

---

## 🔧 Future Improvements

- Hybrid retrieval (BM25 + embeddings)
- Expanded medical datasets
- Improved evaluation model
- Token optimization for scalability

---

## 🎯 Key Learnings

- Agent-based pipelines outperform linear LLM calls
- Retrieval grounding is critical for trust in medical AI
- Self-evaluation adds a measurable reliability layer

---

## 📌 Conclusion

MedBot demonstrates how agentic AI systems can deliver **reliable, context-aware medical assistance**. By combining structured reasoning, retrieval, and evaluation, it moves beyond traditional chatbots toward **trustworthy AI systems**.

---

## ⚡ Tech Stack

- LangGraph
- ChromaDB
- Streamlit
- LLM (RAG-based pipeline)

---

## 🧩 Project Status

✅ Functional  
📊 Evaluated  
🚀 Ready for extension
