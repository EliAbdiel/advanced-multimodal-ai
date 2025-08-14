from langgraph.graph import MessagesState
from src.utils.llm_setup import get_gemini_llm
from src.agents.image_generation import generate_image
from src.agents.link_scraping import scrape_link
from src.agents.deep_search import deep_research_report
from src.agents.conversational_ai import general_question_answer
from src.agents.youtube_transcription import youtube_transcribe


async def get_agent_tools() -> list:
    return [
        generate_image,
        scrape_link,
        deep_research_report,
        general_question_answer,
        youtube_transcribe
    ]

async def get_model_with_tools():
    llm = await get_gemini_llm()
    # Tool binding
    tools = await get_agent_tools()
    return llm.bind_tools(tools, parallel_tool_calls=False)

# Node
async def supervisor_agent(state: MessagesState):
   model_with_tools = await get_model_with_tools()
   message = await model_with_tools.ainvoke(state["messages"])
   return {"messages": [message]}