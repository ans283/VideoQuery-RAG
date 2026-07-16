import os
from dotenv import load_dotenv
import uuid

# LangChain tools for loading, splitting, and storing
from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
# LangChain tools for Gemini (Embeddings and the LLM itself)
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

from operator import itemgetter

# part-1: Load environment variables
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# part-2: Initialize the Gemini embedding model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")



# Phase 1: Load, chunk and store

video_url = input("Enter the YouTube video URL: ")

# 1. Extract the unique 11-character video ID
video_id = video_url.split("v=")[1][:11]

# 2. Create a dynamic folder name for this specific video
db_dir = f"./database/chroma_db_{video_id}"

# 3. Check if THIS specific folder already exists
if os.path.exists(db_dir):
    print(f"Found existing database for video {video_id} 📁. Loading it now...")
    vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings)
else:
    print(f"No database found for {video_id}. Downloading transcript and building it now... ⏳")
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False, language=["en", "en-US", "en-IN"])
    transcript = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(transcript)

    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=db_dir
    )
    print("Database built and saved! ✅")

# ==========================================
# PHASE 2: Retrieve and Generate
# ==========================================

# Set up the strict instructions for the LLM
template = """
Answer the question based on the context below.
Context: {context}
Chat History: {chat_history}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Turn the database into a search engine (retriever)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})

# Helper function to stitch retrieved text chunks into one string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Assemble the RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
        # We need to route the chat history here too!
    }
    | prompt
    | llm
    | StrOutputParser()
)

# Create a dynamic variable for the session
current_session_id = f"session_{str(uuid.uuid4())}"


# Initialize the LangChain SQLite memory tool
chat_memory = SQLChatMessageHistory(
    session_id=current_session_id,
    connection="sqlite:///chat_memory.db"
)
# Phase 3: Interactive Chat Loop

print("\n🤖 Chat with the video! (Type 'quit' to exit)")

while True:
    query = input("\nYour question: ")
    
    if query.lower() == "clear":
        chat_memory.clear()
        print("Chat history cleared.")
        continue

    if query.lower() in ["exit", "quit"]:
        break
    # Retrieve the chat history from the memory
    chat_history = chat_memory.messages

    print("Current memory:", chat_memory.messages)

    result = rag_chain.invoke({"question": query, "chat_history": chat_history})

    ai_answer = result
    # response = rag_chain.invoke(query)

    print(f"AI: {ai_answer}") 

    # Package the turn as a tuple and append it to the memory log
    history_tuple = (query, ai_answer)
    chat_memory.add_user_message(query)
    chat_memory.add_ai_message(ai_answer)