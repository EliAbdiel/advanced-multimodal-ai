import chainlit as cl

from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from src.utils.helpers import safe_json_loads
from src.utils.prompts import (
    generate_youtube_transcribe_prompt, 
    generate_context_and_url_prompt
)
from src.utils.llm_setup import get_gemini_llm_for_youtube, get_gemini_llm_v2

PROMPT = generate_youtube_transcribe_prompt()

EXTRACT_CONTEXT_URL = ChatPromptTemplate.from_messages(
    [
        (
            "user", generate_context_and_url_prompt()
        )
    ]
)

@tool
async def youtube_transcribe(user_message: str):
    """
    Transcribes YouTube videos using Gemini AI with intelligent context extraction.
    
    This tool:
    1. Intelligently extracts YouTube URLs and context from user messages
    2. Validates YouTube URLs
    3. Uses Gemini's multimodal capabilities for transcription
    4. Implements robust JSON error recovery
    5. Handles API errors gracefully
    
    Workflow:
    - First extracts context and URL from the user message using Gemini
    - Validates the extracted YouTube URL
    - If no valid URL found, returns an error message
    - Sends video URL and context to Gemini for transcription
    - Returns cleaned transcription text
    
    Args:
    -----------
    user_message : str
        User input containing YouTube URL and optional context/instructions
    
    Returns:
    --------
    str
        Video transcription text or error message
    
    Example:
    --------
    Input: "Summarize this video about AI safety: https://youtu.be/abc123"
    Output: "In this lecture, Professor Smith discusses three key principles of AI safety..."
    
    Error Handling:
    --------------
    - Returns clear error for invalid/missing URLs
    - Implements JSON repair mechanism for extraction failures
    - Catches and reports API exceptions
    """
    await cl.Message(content="Transcribe YouTube video Selected!\nPlease hold on while I work on it!").send()

    extracted_data = await extract_context_and_url(user_message)
    if len(extracted_data) == 2:
        context, url = extracted_data
    else:
        url = extracted_data[0]
        if not url.startswith("https://www.youtube.com/watch?v=") or not url.startswith("https://youtu.be/"):
            return "No YouTube URL provided. Please check the URL and try again."
        context = PROMPT
        
    try:
        response = await get_gemini_llm_for_youtube(url, context)

        if response.text:
            print("\nTranscription successful!")
            print("\nYouTube Transcription Metadata:")
            print(response.usage_metadata)
        
        return [response.text, url]

    except Exception as e:
        print(f"An error occurred while processing the YouTube video: {e}")
        return "Error processing the video (e.g., malformed URL). Please check the URL and try again."

async def extract_context_and_url(user_message: str):
    """
    This function extracts context and URL from the user message using the Gemini model.
    """
    gemini_llm_v2 = await get_gemini_llm_v2()
    context_url = await gemini_llm_v2.ainvoke(EXTRACT_CONTEXT_URL.format(input=user_message))
    print(f"\nExtracted context and URL: {context_url.content}")
    json_data = await safe_json_loads(context_url.content.strip())
    # print(f"\nParsed JSON data: {json_data}")
    # if len(json_data) == 2:
    #     print(f"\nJSON data context: {json_data[0]}")
    #     print(f"\nJSON data url: {json_data[1]}")
    # else:
    #     print(f"\nJSON data url: {json_data[0]}")
    return json_data

# user_message = "https://youtu.be/dQw4w9WgXcQ?si=123456789 resume el video de youtube en 3 lineas"
# if __name__ == "__main__":
#   asyncio.run(extract_context_and_url(user_message))