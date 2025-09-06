# main.py
from fastapi import FastAPI
from app.routes import router

app = FastAPI(title="RAG API")
app.include_router(router, prefix="/api")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
