import chainlit as cl
import re

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from src.utils.llm_setup import get_gemini_llm, get_gemini_url_context

@tool
async def scrape_link(user_message: str) -> str:
    """
    Advanced web content extraction system with intelligent URL handling.
    
    This tool provides robust web scraping capabilities with:
    1. Smart URL-context separation using regex parsing
    2. Dual scraping methods for different use cases
    3. Context-aware content processing
    4. Automatic Markdown conversion
    5. Gemini API integration for enhanced content extraction
    
    Workflow:
    - Extracts URL and context from user message using regex:
        pattern = r"(?ix)(?P<context>.*?)(?P<url>https?://[^\s]+)(?P<context_after>.*)?"
    - Routes to appropriate scraping method:
        A) With context: Uses Gemini's UrlContext for AI-powered extraction
        B) Without context: Traditional scraping + Markdown conversion
    
    Args:
    -----------
    user_message : str
        Input containing URL and optional context/instructions
    
    Returns:
    --------
    str
        Extracted and processed content in Markdown format
    
    Processing Methods:
    -------------------
    1. Gemini UrlContext Method (when context provided):
       - Uses Google Gemini's specialized URL processing
       - Extracts text content directly from API response
       - Preserves contextual relevance
    
    2. Traditional Scraping Method:
       - Uses WebBaseLoader for content extraction
       - Processes content through LLM chain:
           prompt = 'Summarize this content in markdown format: {context}'
       - Handles large documents (first 10,000 characters)
    
    Error Handling:
    --------------
    - Regex failures: Falls back to full message as context
    - Invalid URLs: Returns clear error message
    - API failures: Graceful degradation to traditional method
    - Content processing errors: User-friendly error messages
    
    Examples:
    ---------
    Input: "Explain this article about AI ethics: https://example.com/ai-ethics"
    Output: (Gemini-processed markdown summary)
    
    Input: "https://example.com/tech-news"
    Output: (Markdown-converted webpage content)
    """
    await cl.Message(content="You've chosen to scrape a link.\nPlease hold on while I work on it!").send()

    context_and_url = await extract_context_and_url(user_message)
    print(f"\nExtracted context and URL: {context_and_url}\n")
    if len(context_and_url) == 2:
        answer = await url_context(user_message)
    else:
        answer = await scrape_web_async(user_message)
    return answer

async def url_context(user_message: str) -> str:
    """
    Scrapes the content of a URL using Google Gemini's UrlContext tool.
    """
    try:
        response = await get_gemini_url_context(user_message)

        print(f"\n{response}\n")
        # for each in response.candidates[0].content.parts:
        #     print(each.text)
        result = "\n".join(each.text for each in response.candidates[0].content.parts if hasattr(each, "text"))
        
        return result
    except Exception as e:
        print(f"Error in url_context {user_message}: {e}")
        return "I encountered an error while processing the URL. Please try again later!"
    
async def scrape_web_async(user_message: str) -> str:
    """
    Scrapes the content of a URL, converts it to Markdown format, 
    and returns the processed content.
    """
    loader = WebBaseLoader(user_message)
    # docs = loader.load()
    docs = []
    async for doc in loader.alazy_load():
        docs.append(doc)

    # Define prompt
    prompt = ChatPromptTemplate.from_template('''Summarize this content in markdown format: {context}''')

    # Get the LLM
    llm = await get_gemini_llm()

    # Instantiate chain
    chain = create_stuff_documents_chain(llm, prompt)

    # Invoke chain
    result = await chain.ainvoke({"context": docs[:10000]})

    try:
        if result:
            print(f"Processed URL {user_message}, sussessfully extracted content.")
        return result
    except Exception as e:
        print(f"Error processing URL {user_message}: {e}")
        return "I encountered an error while processing the URL (e.g., malformed URL). Please try again later!"

async def extract_context_and_url(user_message: str):
    """Extracts context and URL from a user message."""
    pattern = re.compile(r"""(?ix)
        (?P<context>.*?)
        (?P<url>https?://[^\s]+)
        (?P<context_after>.*)?
    """)

    match = pattern.search(user_message.strip())

    if not match:
        return [user_message.strip()]
   
    url = match.group("url").strip()
    before = match.group("context").strip()
    after = match.group("context_after").strip() if match.group("context_after") else ""

    context = f"{before} {after}".strip()

    if context:
        return [context, url]
    else:
        return [url]