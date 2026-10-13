import streamlit as st
from backend.rag_engine import setup_rag_system

# 1. Page Setup
st.title("🤖 YouTube RAG Chatbot")
video_url = st.text_input("Enter the YouTube video URL:")

# 2. Setup Phase
if video_url:
    st.info("Loading the AI brain... ⏳")
    rag_chain = setup_rag_system(video_url)
    st.success("Database built and ready to chat! ✅")
    
    # 3. Chat History Setup
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
    # 4. Chat Interface
    prompt = st.chat_input("Ask a question about the video...")
    
    if prompt:
        # Save and draw user question
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        # Trigger the AI
        response = rag_chain.invoke({
            "input": prompt,
            "chat_history": st.session_state.messages
        })
        
        # Save and draw AI answer
        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.write(response)