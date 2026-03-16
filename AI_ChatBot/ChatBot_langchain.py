
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from functools import lru_cache
from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate

load_dotenv()

app = FastAPI(title="AI YouTube Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Models
# -----------------------------

class ChatRequest(BaseModel):
    message: str
    video_url: str


# -----------------------------
# LLM
# -----------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant"
)

# -----------------------------
# Embeddings
# -----------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# Text Splitter
# -----------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the provided transcript context.
Keep the answer short and correct.

If the context is insufficient or the question asked is vague or completely does not
match with any of the aspects of video , say "this question is out of scope of this video, but i still tell you the answer",
then in atmost 2-3 sentence answer it by searching your database .

Context:
{context}
, and the question asked is 
Question:
{question}
""",
    input_variables=["context", "question"]
)


# -----------------------------
# Transcript Fetching
# ----------------------------


# -----------------------------
# Vector Store Cache
# -----------------------------

@lru_cache(maxsize=20)
def build_vector_store(video_url: str):
    try:
        loader = YoutubeLoader.from_youtube_url(video_url,
           add_video_info=False,
           language="en"
        )

        docs = loader.load()

        split_docs = splitter.split_documents(docs)
        if not split_docs:
            raise HTTPException(status_code=500, detail="Failed to split transcript into chunks")
        
        vector_store = Chroma.from_documents(
            split_docs,
            embedding_model
        )
    
        return vector_store
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Error fetching transcript: {str(e)}")


# -----------------------------
# Chat Endpoint
# -----------------------------

@app.post("/chat")
async def chat(request: ChatRequest):

    if not request.message.strip():
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty"
        )

   # Get cached vector store
    vector_store = await run_in_threadpool(
        build_vector_store,
        request.video_url
    )

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    retrieved_docs = await run_in_threadpool(
        retriever.invoke,
        request.message
    )

    context_text = "\n\n".join(
        doc.page_content for doc in retrieved_docs
    )

    final_prompt = prompt.format(
        context=context_text,
        question=request.message
    )

    try:
        answer = await run_in_threadpool(llm.invoke, final_prompt)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    return {"response": answer.content}
