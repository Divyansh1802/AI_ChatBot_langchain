import asyncio

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi,TranscriptsDisabled
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AI Chatbot API")

from dotenv import load_dotenv
load_dotenv()

# Request model
class ChatRequest(BaseModel):
    message: str
    video_id: str
    
# API Endpoint
@app.post("/chat")
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    response = await generate_response(request)

    return {"response": response}


llm = ChatGroq(
    model="llama-3.1-8b-instant"
)    
    
embedding_model =  HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2" )

prompt = PromptTemplate(
    template="""
      You are a helpful assistant,
      answer in very short and concised manner Only form the provided transcript context.
      If the context is insufficient , just say you don't know.
      {context}
      Question: {question}
    """,
    input_variables=['context','question']
    )

async def generate_response(request: ChatRequest) -> str:
    
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(request.video_id)
        transcript = " ".join(chunk["text"] for chunk in transcript_list)
    
    except TranscriptsDisabled:
           raise HTTPException(status_code=404, detail="ERROR")
    
    
    # SPLIITER

    splitter = RecursiveCharacterTextSplitter(chunk_size = 1000,chunk_overlap = 200)

    chunks = splitter.create_documents([transcript])

    #EMBEDDING GENERATION AND STORING IT IN VECTOR STORE
    
    vector_store =  FAISS.from_documents(chunks,embedding_model)


    # RETRIEVAL

    retriever = vector_store.as_retriever(search_type = "similarity", 
                                       search_kwargs = {"k":4})
    
    
    retrieved_docs = retriever.invoke(request.message)
    
    context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
    
    final_prompt = prompt.invoke({"context": context_text,"question":request.message})
    
    answer = llm.invoke(final_prompt)
    
    return answer.content
