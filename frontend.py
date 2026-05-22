import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG Chatbot", layout="wide")

st.title("RAG Chatbot")

# Upload PDF
uploaded_file = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    files = {
        "file": uploaded_file.getvalue()
    }

    response = requests.post(
        f"{API_URL}/upload",
        files={
            "file": (
                uploaded_file.name,
                uploaded_file,
                "application/pdf"
            )
        }
    )

    if response.status_code == 200:
        st.sidebar.success("PDF uploaded successfully")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
prompt = st.chat_input("Ask a question")

if prompt:

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    response = requests.post(
        f"{API_URL}/ask",
        json={"question": prompt}
    )

    data = response.json()
    if "answer" in data:
        answer = data["answer"]
    else:
        answer = f"Error: {data}"

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    with st.chat_message("assistant"):
        st.markdown(answer)

        # Sources
        if "sources" in data:

            with st.expander("Sources"):

                for source in data["sources"]:
                    st.write(source)