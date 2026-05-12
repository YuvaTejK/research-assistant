import warnings
warnings.filterwarnings("ignore")
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.prebuilt import create_react_agent
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ── Page Config ──────────────────────────────────────────────────
st.set_page_config(page_title="AI Research Assistant", page_icon="🔍")
st.title("🔍 AI Research Assistant")
st.caption("Powered by Groq + LLaMA 3 — Ask me anything!")

# ── Sidebar: PDF Upload ──────────────────────────────────────────
st.sidebar.title("📄 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file is not None:
    with st.sidebar:
        with st.spinner("Processing PDF..."):
            os.makedirs("data", exist_ok=True)
            temp_path = f"data/{uploaded_file.name}"
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(temp_path)
            documents = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=512, chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            if os.path.exists("faiss_index"):
                vectorstore = FAISS.load_local(
                    "faiss_index", embeddings,
                    allow_dangerous_deserialization=True
                )
                vectorstore.add_documents(chunks)
            else:
                vectorstore = FAISS.from_documents(chunks, embeddings)

            vectorstore.save_local("faiss_index")
            st.cache_resource.clear()
            st.success(f"✅ {uploaded_file.name} indexed!")

st.sidebar.markdown("---")
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state.messages = []
    st.rerun()

# ── Agent Loader ─────────────────────────────────────────────────
@st.cache_resource
def load_agent():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

    tools = []

    try:
        web_tool = TavilySearchResults(max_results=2)
        tools.append(web_tool)
        print("✅ Web search tool loaded")
    except Exception as e:
        print(f"⚠️ Web search not available: {e}")

    if os.path.exists("faiss_index"):
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectorstore = FAISS.load_local(
            "faiss_index", embeddings,
            allow_dangerous_deserialization=True
        )
        retriever_tool = create_retriever_tool(
            vectorstore.as_retriever(search_kwargs={"k": 3}),
            name="document_search",
            description="Search through uploaded PDF documents for information"
        )
        tools.append(retriever_tool)
        print("✅ Document search tool loaded")

    system_prompt = """You are a helpful research assistant.
Use tools to find accurate information. Be concise and clear.
Always cite your sources."""

    agent = create_react_agent(model=llm, tools=tools, prompt=system_prompt)
    return agent


# ── Chat State ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_request_time" not in st.session_state:
    st.session_state.last_request_time = 0

# ── Show Chat History ────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat Input ───────────────────────────────────────────────────
if user_input := st.chat_input("Ask me anything..."):

    time_since_last = time.time() - st.session_state.last_request_time
    if time_since_last < 3:
        st.info("⏳ Please wait a moment before sending another message.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    history = []
    for msg in st.session_state.messages[-7:-1]:
        if msg["role"] == "user":
            history.append(HumanMessage(content=msg["content"]))
        else:
            history.append(AIMessage(content=msg["content"]))

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                agent = load_agent()
                response = agent.invoke({
                    "messages": history + [HumanMessage(content=user_input)]
                })

                # Show tools used
                tool_calls = []
                for msg in response["messages"]:
                    if hasattr(msg, "tool_calls") and msg.tool_calls:
                        for call in msg.tool_calls:
                            tool_calls.append(f"🔧 **{call['name']}**")

                if tool_calls:
                    with st.expander("🔧 Tools Used", expanded=False):
                        for call in tool_calls:
                            st.markdown(call)

                answer = response["messages"][-1].content
                st.write(answer)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                st.session_state.last_request_time = time.time()

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")