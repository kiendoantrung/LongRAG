from fastapi import APIRouter, Request
from .models import QueryRequest, QueryResponse

router = APIRouter()

@router.post("/query", response_model=QueryResponse)
async def process_query(request: Request, query_request: QueryRequest):
    workflow = request.app.state.workflow
    query_engine = request.app.state.query_engine
    result = await workflow.run(
        query_str=query_request.query,
        query_eng=query_engine,
    )
    return QueryResponse(response=str(result))