# 🔍 AI Research Assistant

A production-ready Multi-Agent AI Research Assistant built with LangChain, LangGraph, Groq (LLaMA 3), FAISS, and Streamlit.

## 🚀 Features
- 💬 Conversational AI with memory (last 6 messages)
- 🌐 Real-time web search using Tavily API
- 📄 PDF upload and intelligent Q&A (RAG)
- ⚡ Streaming responses word by word
- 🔧 Shows which tools the agent used

## 🛠️ Tech Stack
| Component | Technology |
|---|---|
| LLM | Groq (LLaMA 3 8B) |
| Agent Framework | LangChain + LangGraph |
| Vector Store | FAISS |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Web Search | Tavily API |
| Frontend | Streamlit |

## ⚙️ How to Run Locally

1. Clone the repo
   git clone https://github.com/YuvaTejK/research-assistant.git
   cd research-assistant

2. Install dependencies
   pip install -r requirements.txt

3. Create .env file
   GROQ_API_KEY=your_groq_key_here
   TAVILY_API_KEY=your_tavily_key_here

4. Run the app
   streamlit run app.py

## 🏗️ Architecture
User Query → LangGraph Agent → Decision
                                    ├── Web Search (Tavily)
                                    ├── Document Search (FAISS + RAG)
                                    └── Direct Answer (LLaMA 3)
                               → Streamed Response

## 📁 Project Structure
research-assistant/
├── app.py          # Main Streamlit app + Agent
├── ingest.py       # PDF ingestion pipeline
├── requirements.txt
└── .gitignore