import os
import asyncio
import json

from dotenv import load_dotenv
from google import genai
from google.genai import types
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from prompts import generate_context_and_url_prompt, generate_youtube_transcribe_prompt
from langsmith import traceable

# Load environment variables from .env file
load_dotenv(override=True)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=0.1,  # <-- Super slow! We can only make a request once every 10 seconds!!
    check_every_n_seconds=0.1,  # Wake up every 100 ms to check whether allowed to make a request,
    max_bucket_size=10,  # Controls the maximum burst size.
)

llm_google = ChatGoogleGenerativeAI(
    model=os.environ["GEMINI_MODEL"],
    rate_limiter=rate_limiter,
    google_api_key=os.environ["GEMINI_API_KEY"],
    temperature=0.3,
)

llm_google_v2 = ChatGoogleGenerativeAI(
    model=os.environ["GEMINI_MODEL"],
    rate_limiter=rate_limiter,
    google_api_key=os.environ["GEMINI_API_KEY_V2"],
    temperature=0.3,
)

client = genai.Client(api_key=os.environ["NAHOMY_GEMINI_KEY"])

EXTRACT_CONTEXT_URL = ChatPromptTemplate.from_messages(
    [
        (
            "user", generate_context_and_url_prompt()
        )
    ]
)

async def extract_context_and_url(user_message: str):
    """
    This function extracts context and URL from the user message using the Gemini model.
    """
    context_url = await llm_google_v2.ainvoke(EXTRACT_CONTEXT_URL.format(input=user_message))
    print(f"\nExtracted context and URL: {context_url.content}")
    json_data = await safe_json_loads(context_url.content.strip())
    # print(f"\nParsed JSON data: {json_data}")
    # if len(json_data) == 2:
    #     print(f"\nJSON data context: {json_data[0]}")
    #     print(f"\nJSON data url: {json_data[1]}")
    # else:
    #     print(f"\nJSON data url: {json_data[0]}")
    return json_data

async def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse failed: {e}")
        try:
            fixed_json = await llm_google.ainvoke(f'Fix this JSON and return only the corrected format ["context_of_the_request", "url_from_input"]: {text}')
            return json.loads(fixed_json.content.strip())
        except Exception as fix_error:
            print(f"Failed to fix JSON: {fix_error}")
            return []

PROMPT = generate_youtube_transcribe_prompt()

@traceable
async def youtube_transcribe(user_message: str):
    """
    This function takes a YouTube video URL as input and uses the Gemini model to:
    1. Transcribe the audio content from the video
    2. Generate timestamps for key events
    3. Provide visual descriptions of what's happening in the video
    
    Args:
        url (str): The URL of the YouTube video to analyze
        
    Returns:
        str: A text response containing the transcription, timestamps and visual descriptions
    """
    extracted_data = await extract_context_and_url(user_message)
    if len(extracted_data) == 2:
        context, url = extracted_data
    else:
        url = extracted_data[0]
        if not url.startswith("https://www.youtube.com/watch?v=") or not url.startswith("https://youtu.be/"):
            return "No YouTube URL provided. Please check the URL and try again."
        context = PROMPT
        
    try:
        response = client.models.generate_content(
            model=os.environ["GEMINI_2_5_MODEL"],
            contents=types.Content(
                parts=[
                    types.Part(
                        file_data=types.FileData(file_uri=url)
                    ),
                    types.Part(text=context)
                ]
            )
        )

        if response.text:
            print("\nTranscription successful!")
            print("\nYouTube Transcription Metadata:")
            print(response.usage_metadata)
        
        return response.text, url

    except Exception as e:
        print(f"An error occurred while processing the YouTube video: {e}")
        return "Error processing the video (e.g., malformed URL). Please check the URL and try again."


# user_message = "https://youtu.be/dQw4w9WgXcQ?si=123456789 resume el video de youtube en 3 lineas"
# if __name__ == "__main__":
#   asyncio.run(extract_context_and_url(user_message))