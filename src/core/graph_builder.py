from langgraph.prebuilt import ToolNode
from langgraph.graph import MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import tools_condition
from langgraph.graph import START, END, StateGraph
from .supervisor import supervisor_agent, get_agent_tools

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

async def build_graph():
    """Build and return the agent workflow graph with memory."""
    # use persistent memory to save conversation history
    # TODO: be compatible with SQLite / PostgreSQL
    memory = MemorySaver()

    # build state graph
    builder = await _build_base_graph()
    
    return builder.compile(checkpointer=memory)