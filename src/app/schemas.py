from pydantic import BaseModel, Field


class ResponseSchema(BaseModel):
    input: str =  Field(..., description="User given input")
    output: str =  Field(..., description="LLM generated output")
    is_rag: bool =  Field(..., description="True or False based whether LLM Generated or retrieved from vector data base using agent tool")