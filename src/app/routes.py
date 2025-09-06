# app/routes.py
from fastapi import APIRouter
from app.agent import run_agent_chain
import asyncio

router = APIRouter()

@router.get("/query")
async def query_agent(q: str):
    """
    Example: GET /query?q=Who are all the passengers in the Journey?
    """
    # Async safe: run blocking code in threadpool
    result = await asyncio.to_thread(run_agent_chain, q)
    return result
