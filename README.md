# 📄 Chat with PDF – Ask Questions from Any PDF using LLMs

Interact with your PDF files using natural language! This app lets you upload a PDF, ask questions about it, and receive intelligent answers powered by **LLMs** and **RAG (Retrieval-Augmented Generation)**.

Live Demo 👉 [chat-with-pdf-project.streamlit.app](https://chat-with-pdf-project.streamlit.app)

---

## 🔍 Features

- 📎 Upload any PDF
- 💬 Ask questions in plain English
- ⚙️ Powered by `LLaMA3` via **Groq API**
- 🧠 Embedding with `HuggingFace MiniLM`
- 🔎 Smart context retrieval using `LangChain`
- 🌐 Fully deployed on **Streamlit Cloud**

---

## ⚙️ Tech Stack

| Tool           | Purpose                          |
|----------------|----------------------------------|
| **Streamlit**  | UI / Deployment                  |
| **LangChain**  | RAG pipeline & chaining          |
| **Groq API**   | LLaMA3 inference (low latency)   |
| **HuggingFace**| Embedding & semantic search      |
| **PyPDFLoader**| PDF parsing and loading          |

---

## 🧪 Example Use Cases

> ✅ Upload a **research paper** and ask:  
> 💬 _“Summarize the findings of section 3”_

> ✅ Upload a **resume** and ask:  
> 💬 _“What are this candidate’s skills?”_

> ✅ Upload a **manual or safety report** and ask:  
> 💬 _“What are the key safety protocols?”_
