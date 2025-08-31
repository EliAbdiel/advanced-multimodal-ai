import asyncio
import platform

from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
# from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, END, StateGraph
from .supervisor import supervisor_agent, get_agent_tools
from src.utils.config import MEMORY_DATABASE 

# Set the correct event loop policy for Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(
        asyncio.WindowsSelectorEventLoopPolicy()
    )

async def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    tools = await get_agent_tools()

    # Graph
    builder = StateGraph(MessagesState)

    # Define nodes: these do the work
    builder.add_node("supervisor", supervisor_agent)
    builder.add_node("tools", ToolNode(tools))

    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges(
        "supervisor",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    builder.add_edge("tools", END)

    return builder

async def memory_saver():
    """Set up a PostgreSQL memory saver for persistent conversation history.
    
    Returns:
        AsyncPostgresSaver: Configured PostgreSQL checkpointer for storing conversation state.
    """
    async with AsyncPostgresSaver.from_conn_string(MEMORY_DATABASE) as checkpointer:
        return await checkpointer.setup()

async def build_graph():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    checkpointer = await memory_saver()

    # checkpointer = InMemorySaver()

    # build state graph
    builder = await _build_base_graph()
    
    return builder.compile(checkpointer=checkpointer)