import os
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

def setup_rag_system(video_url):

    # 1. Initialize models
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")
    
    # 2. Build or load database
    video_id = video_url.split("v=")[1][:11]
    db_dir = f"./database/chroma_db_{video_id}"
    
    if os.path.exists(db_dir):
        vector_store = Chroma(persist_directory=db_dir, embedding_function=embeddings)
    else:
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
        transcript = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(transcript)
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_dir)

        
    # 3. Create retrievers and chains
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Given a chat history and the latest user question, rewrite it into a standalone question. If it makes sense on its own, return it as is."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Use the context to answer. If you don't know, say so.\n\nContext: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])
    
    rag_chain = (
        {
            "context": history_aware_retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)), 
            "input": itemgetter("input"),
            "chat_history": itemgetter("chat_history"),
        }
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain