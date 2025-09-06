# app/agent.py
import time


from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from app.schemas import ResponseSchema
from app.log import logger
from app.vectors import load_vector


retriever = load_vector()

def get_le_horla_story_docs(query: str, vector_db_retriever: BaseRetriever) -> str:
    retrieved_docs = vector_db_retriever.get_relevant_documents(query)
    if not retrieved_docs:
        return "No relevant documents found."
    context = "\n\n".join([page.page_content for page in retrieved_docs])
    return f"Answer based on context: {context}"

def get_le_horla_story_docs_wrapper(input: str) -> str:
    return get_le_horla_story_docs(query=input, vector_db_retriever=retriever)

rag_tool = Tool(
    name="get_le_horla_story_docs",
    description="ONLY use this tool to answer questions about 'Le Horla' story.",
    func=get_le_horla_story_docs_wrapper
)

llm = ChatOpenAI(model="gpt-4o-mini")


agent = initialize_agent(
    tools=[rag_tool],
    llm=llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)

schema = PydanticOutputParser(pydantic_object=ResponseSchema)
format_instruction = schema.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Respond to user queries by thinking and analyzing as per user instructions"),
    ("user", "Question: {input} format instruction {format_instruction}")
])

# -----------------------
# Run Agent with retry
# -----------------------
def run_agent_chain(user_input: str, max_retries: int = 3, retry_delay: float = 1.0):
    def wrap_agent_output(agent_result: dict) -> str:
        return agent_result["output"]


    for attempt in range(max_retries):
        try:
            start=  time.perf_counter()
            chain = prompt | agent | RunnableLambda(wrap_agent_output) | JsonOutputParser()
            response: dict = chain.invoke({"input": user_input, "format_instruction": format_instruction})
            response.update({"inference_delay": time.perf_counter() - start})
            return response

        except Exception as e:
            logger.warning(f"Attempt {attempt+1} failed: {e}")
            time.sleep(retry_delay)

    return ResponseSchema(
        input=user_input,
        output="Failed to generate a response after retries.",
        is_rag=False
    )
