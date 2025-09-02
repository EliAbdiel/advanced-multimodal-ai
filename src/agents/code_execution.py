import chainlit as cl

from google.genai import types
from google.genai.types import GenerateContentResponse
from src.utils.llm_setup import get_gemini_client, get_llm_coder
from src.utils.config import GEMINI_2_5_MODEL
from langchain_core.tools import tool

@tool
async def code_generation(user_message: str) -> str:
    """
    Generates and executes code snippets to solve programming-related tasks.

    This tool leverages the Gemini LLM with code execution capabilities to
    produce high-quality, runnable code in response to natural language queries.
    It not only generates the code but also executes it and returns the results,
    making it suitable for computational tasks, algorithm exploration, and
    rapid prototyping.

    Workflow:
    1. Initializes Gemini client with code execution enabled.
    2. Sends the user's request along with system instructions.
    3. Receives model-generated response containing:
       - Explanation text
       - Generated code
       - Executed output
    4. Formats and returns the combined result.

    Args:
    -------
    user_message : str
        A natural language request describing the coding task or problem.

    Returns:
    --------
    str
        A formatted response including explanation, generated code, and execution output.

    Use Cases:
    ----------
    - Solving algorithmic or math problems via code
    - Demonstrating coding patterns and examples
    - Running quick data processing tasks
    - Exploring multi-language code generation with execution

    Example:
    --------
    >>> await code_generation("Calculate the sum of the first 50 prime numbers")
    "The sum of the first 50 prime numbers is 5117.

    ```python
    # Python code to calculate sum of first 50 primes
    def is_prime(n):
        ...
    print(sum_primes(50))
    ```

    5117"
    """
    await cl.Message(content="Code execution Selected!\nPlease wait while I work on it!").send()

    coder = await get_llm_coder()

    response = await coder.ainvoke(user_message)

    return response.content

    # input = ("What is the sum of the first 50 prime numbers? "
    #         "Generate and run code for the calculation, and make sure you get all 50.")

    # system_instruction = """
    #         You are an expert software developer and a helpful coding assistant.
    #         You are able to generate high-quality code in any programming language.
    #     """

    # chat = client.chats.create(
    #     model=GEMINI_2_5_MODEL,
    #     config=types.GenerateContentConfig(
    #         system_instruction=system_instruction,
    #         tools=[types.Tool(code_execution=types.ToolCodeExecution)]
    #     ),
    # )

    # response = chat.send_message(user_message)

    # return await get_response(response)

# async def get_response(response: GenerateContentResponse) -> str:
#     """Processes the response from the Gemini model to extract text, code, and output."""
#     text = ""
#     code = ""
#     output = ""

#     for part in response.candidates[0].content.parts:
#         if part.text is not None:
#             text= part.text
#         if part.executable_code is not None:
#             code= part.executable_code.code
#         if part.code_execution_result is not None:
#             output = part.code_execution_result.output

#     answer = f"{text}\n\n```python\n{code}\n```\n\n{output}"
#     print(f"\n{answer}\n")
#     return answer