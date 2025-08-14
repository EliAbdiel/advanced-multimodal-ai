from src.utils.llm_setup import get_openrouter_llm
from langchain_core.tools import tool

@tool
async def general_question_answer(user_message: str):
    """
    General-purpose question answering using OpenRouter's language models.
    
    This tool provides:
    - Direct access to advanced LLMs via OpenRouter
    - Zero-shot question answering capabilities
    - Minimal-latency response generation
    - Stateless API interaction
    
    Workflow:
    1. Initializes OpenRouter LLM instance
    2. Directly invokes model with user query
    3. Returns raw model response
    
    Args:
    -----------
    user_message : str
        Natural language question or prompt
    
    Returns:
    --------
    str
        Model-generated response text
    
    Technical Details:
    -----------------
    - Uses OpenRouter API for model access
    - Maintains minimal configuration overhead
    - Leverages async I/O for high concurrency
    
    Ideal For:
    ---------
    - General knowledge questions
    - Conceptual explanations
    - Brief factual responses
    - Low-complexity reasoning tasks
    
    Example Usage:
    -------------
    Input: "Explain quantum entanglement in simple terms"
    Output: "Quantum entanglement is a phenomenon where two particles..."
    """
    model = await get_openrouter_llm()
    response = await model.ainvoke(user_message)
    return response.content