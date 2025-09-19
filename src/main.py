import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router
from api.config import get_settings, get_llm
from contextlib import asynccontextmanager
from workflow.long_rag import LongRAGWorkflow
from constants import DEFAULT_CHUNK_SIZE, DEFAULT_TOP_K, DEFAULT_SMALL_CHUNK_SIZE

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    llm = get_llm()
    
    workflow = LongRAGWorkflow(timeout=60)
    result = await workflow.run(
        data_dir=settings.data_dir,
        llm=llm,
        chunk_size=DEFAULT_CHUNK_SIZE,
        similarity_top_k=DEFAULT_TOP_K,
        small_chunk_size=DEFAULT_SMALL_CHUNK_SIZE,
    )
    app.state.workflow = workflow
    app.state.query_engine = result["query_engine"]
    
    yield

app = FastAPI(title="LongRAG API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    settings = get_settings()
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)