import os
import streamlit as st
import time
import sqlite3
import random
import pandas as pd
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

def get_db_connection():
    db_path = os.path.join(os.getcwd(), "ux_data.db")
    return sqlite3.connect(db_path)
    
load_dotenv()
hf_token = st.secrets["HF_TOKEN"]
DB_FAISS_PATH = "vectorstore/db_faiss"
@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        encode_kwargs={"normalize_embeddings": True}
    )

    db = FAISS.load_local(
        DB_FAISS_PATH,
        embedding_model,
        allow_dangerous_deserialization=True
    )
    return db

def get_prompt():
    return ChatPromptTemplate.from_template("""
Use the pieces of information provided in the context to answer user's question in structured format.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Don't provide anything out of the given context.

Context: {context}
Question: {input}
Answer in a structured format.
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_chain(vectorstore):
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0,
        groq_api_key=st.secrets["GROQ_API_KEY"]
    )

    retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5} # tried k=3 first but it missed context
    )

    prompt = get_prompt()

    chain = (
        {
            "context": retriever | format_docs,
            "input": RunnablePassthrough()
        }
        | prompt
        | llm
    )
    return chain

def log_feedback(mode, latency, speed, frustration, trust, age):
    conn = get_db_connection()
    c = conn.cursor()

    c.execute("""
    INSERT INTO feedback (timestamp, mode, latency, speed, frustration, trust, age)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (time.time(), mode, latency, speed, frustration, trust, age))
    conn.commit()
    conn.close()

def view_data():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM feedback", conn)
    conn.close()
    return df

def init_db():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp REAL,
        mode TEXT,
        latency REAL,
        speed INTEGER,
        frustration INTEGER,
        trust INTEGER,
        age INTEGER
    )
    """)
    conn.commit()
    conn.close()

def tips(user_input):
    return "generate tips according to user inputs Or give info of upcoming Event, Conference, fest."
def main():
    init_db()
    st.image("https://images.careerindia.com/img/2014/03/07-abv-iiitmgwalior.jpg", width=100)
    st.title("IIIT Gwalior Chatbot")
    st.caption("Ask anything about IIIT Gwalior")
    mode = st.radio("Choose waiting experience", [
        "Simple (spinner)",
        "Progress + Steps",
        "Streaming + Tips"
    ])

    # Sidebar Admin Panel
    st.sidebar.title("Admin Panel")
    if "show_data" not in st.session_state:
        st.session_state.show_data = False

    if st.sidebar.button("Show Feedback Data"):
        st.session_state.show_data = True

    if st.session_state.show_data:
        df = view_data()
        st.sidebar.dataframe(df)

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "show_form" not in st.session_state:
        st.session_state.show_form = False
        
    if "last_latency" not in st.session_state:
        st.session_state.last_latency = 0
    if "last_mode" not in st.session_state:
        st.session_state.last_mode = mode

    user_input = st.chat_input("Ask your question")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({"role": "user", "content": user_input})

        try:
            vectorstore = get_vectorstore()
            chain = get_chain(vectorstore)

            start_time = time.time()
            docs = vectorstore.as_retriever(search_kwargs={"k": 5}).invoke(user_input)
            progress = None

            if mode == "Simple (spinner)":
                with st.spinner("Thinking..."):
                    response = chain.invoke(user_input)
                answer = response.content

            elif mode == "Progress + Steps":
                status = st.empty()
                progress = st.progress(0)
                status.info("Understanding your question...")
                progress.progress(30)
                status.info("Fetching relevant information...")
                progress.progress(60)
                response = chain.invoke(user_input)
                answer = response.content
                progress.progress(100)
                status.success("Answer ready")

            else:
                tip = tips(user_input)
                st.info(f"Tip: {tip}")
                response = chain.invoke(user_input)
                answer = response.content

            latency = time.time() - start_time
            st.session_state.last_latency = latency
            st.session_state.last_mode = mode

            with st.chat_message("assistant"):
                placeholder = st.empty()
                typed = ""
                for char in answer:
                    typed += char
                    placeholder.markdown(typed + "|")
                    time.sleep(random.uniform(0.005, 0.02))

                st.caption(f"Response time: {latency:.2f}s")
                st.session_state.show_form = True

                if latency < 2:
                    st.success("Instant response")
                elif latency < 5:
                    st.info("Moderate response")
                else:
                    st.warning("Slow response")

                with st.expander("Sources"):
                    for i, doc in enumerate(docs[:3]):
                        st.markdown(f"**Source {i+1}**")
                        st.caption(doc.metadata.get("source", "Unknown"))
                        st.write(doc.page_content[:200] + "...")
                        st.divider()

            st.session_state.messages.append({"role": "assistant", "content": answer})

            if progress:
                progress.empty()

        except Exception as e:
            st.error(f"Error: {str(e)}")

    # Feedback form 
    if st.session_state.show_form:
        st.markdown("### Give your feedback")
        with st.form("feedback_form"):
            age = st.number_input("Your Age", min_value=10, max_value=100, value=20)
            col1, col2, col3 = st.columns(3)
            with col1:
                speed = st.slider("Speed", 1, 5, 3)
            with col2:
                frustration = st.slider("Frustration", 1, 5, 2)
            with col3:
                trust = st.slider("Trust", 1, 5, 4)
            submit = st.form_submit_button("Submit Feedback")
            if submit:
                log_feedback(
                    st.session_state.last_mode,
                    st.session_state.last_latency,
                    speed, frustration, trust, age
                )
                st.success("Feedback saved!")
                st.session_state.show_form = False
                st.session_state.show_data = True
                st.rerun()  
if __name__ == "__main__":
     main()
    

