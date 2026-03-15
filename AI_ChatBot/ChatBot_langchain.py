from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from functools import lru_cache

from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
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
    video_id: str


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
    chunk_size=1000,
    chunk_overlap=200
)

# -----------------------------
# Prompt
# -----------------------------

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY using the provided transcript context.
Keep the answer very short and concise.

If the context is insufficient say:
"I don't know".

Context:
{context}

Question:
{question}
""",
    input_variables=["context", "question"]
)


# -----------------------------
# Transcript Fetching
# -----------------------------

async def fetch_transcript(video_id: str):

    try:
        transcript_items = await run_in_threadpool(
            YouTubeTranscriptApi.get_transcript,
            video_id,
            ['en']
        )

        transcript = " ".join(
            item["text"].strip() for item in transcript_items
        )

        return transcript

    except TranscriptsDisabled:
        raise HTTPException(status_code=404, detail="Transcript disabled")

    except Exception as e:
        raise HTTPException(
            status_code=404,
            detail=f"Transcript fetch error: {str(e)}"
        )


# -----------------------------
# Vector Store Cache
# -----------------------------

@lru_cache(maxsize=20)
def build_vector_store(video_id: str, transcript: str):

    docs = splitter.create_documents([transcript])

    vector_store = FAISS.from_documents(
        docs,
        embedding_model
    )

    return vector_store


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

    # Fetch transcript
    transcript = await fetch_transcript(request.video_id)

    # Get cached vector store
    vector_store = await run_in_threadpool(
        build_vector_store,
        request.video_id,
        transcript
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

    answer = await run_in_threadpool(
        llm.invoke,
        final_prompt
    )

    return {"response": answer.content}
