import chainlit as cl
import asyncio

from langchain_core.tools import tool
from langchain.prompts import ChatPromptTemplate
from src.utils.llm_setup import get_gemini_llm, get_gemini_llm_v2
from src.utils.prompts import (
    generate_critical_thinker_prompt, 
    generate_research_report_prompt, 
    generate_search_queries_prompt,
    generate_webpage_summary_template,
)
from src.utils.helpers import json_loads
from src.services.search_and_scrape import (
    web_search_with_tavily, 
    scrape_link_async, 
    flatten_list_of_list
)

@tool
async def deep_research_report(user_message: str):
	"""
    Conducts comprehensive web research using a multi-stage pipeline to generate detailed reports.
    
    This tool employs a sophisticated 4-stage research process:
    1. Query Generation: Creates optimized search queries from user questions
    2. Web Search: Executes parallel searches using Tavily API
    3. Content Processing: Scrapes, summarizes, and analyzes web content
    4. Report Synthesis: Compiles findings into a cohesive research report
    
    Key Features:
    - Multi-query generation for comprehensive coverage
    - Parallel processing of search results
    - Intelligent content summarization with LLMs
    - Robust JSON error recovery
    - Context-aware report generation
    
    Workflow:
    1. Generate search queries from user input
    2. Perform parallel web searches
    3. For each URL:
        a. Scrape and sanitize content
        b. Generate AI-powered summary
    4. Compile all summaries into research context
    5. Generate final report using advanced prompt engineering
    
    Args:
    -----------
    user_message : str
        Research question or topic (e.g., "Latest advancements in quantum computing")
    
    Returns:
    --------
    str
        Comprehensive research report in Markdown format
    
    Example Output Structure:
    ------------------------
    # Research Report: [Topic]
    
    ## Key Findings
    - [Summary point 1]
    - [Summary point 2]
    
    ## Detailed Analysis
    [In-depth discussion of findings with sources]
    
    ## Conclusion
    [Synthesized insights and recommendations]
    
    Error Handling:
    --------------
    - Gracefully handles empty search results
    - Recovers from JSON parsing errors via LLM repair
    - Manages content scraping failures
    - Fallback to partial results when individual URLs fail
    """
	await cl.Message(content="Search on the Web Browser Selected!\nPlease wait while I work on it!").send()

	# question = "Encuentra informacion sobre RAG (Retrieval-Augmented Generation)"
	results = await generate_report(user_message)
	# print(f"\n\n{results}\n")
	return results.content

async def generate_report(question):
    """Generates a research report based on the provided question."""
    research_results = await process_search_questions(question)
    context = flatten_list_of_list(research_results)
    gemini_llm_v2 = await get_gemini_llm_v2()
    report = await gemini_llm_v2.ainvoke(prompt.format(context=context, question=question))
    return report

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", generate_critical_thinker_prompt()),
        ("user", generate_research_report_prompt())
    ]
)

async def process_search_questions(question):
    """Processes a search question by generating search queries and scraping results."""
    gemini_llm = await get_gemini_llm()
    search_output = await gemini_llm.ainvoke(SEARCH_PROMPT.format(question=question))
    print(f"\nRaw output from LLM: {search_output.content.strip()}\n")
    queries = await json_loads(search_output.content.strip())
    tasks = [process_search_results({"question": q}) for q in queries]
    return await asyncio.gather(*tasks)

SEARCH_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "user", generate_search_queries_prompt()
        )
    ]
)

async def process_search_results(data):
    """
    Processes search results by scraping and summarizing each URL.
    """
    urls = await web_search_with_tavily(data["question"])
    if not urls:
        print(f"No results could be obtained for the search: {data['question']}")
        return []

    tasks = []
    for url in urls:
        # summary = await scrape_and_summarize({"question": data["question"], "url": url})
        tasks.append(scrape_and_summarize({"question": data["question"], "url": url}))

    try:
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(f"Error processing results: {e}")
        return []
    
async def scrape_and_summarize(url_data):
    """Scrapes the content of a URL and summarizes it using an LLM."""
    text = await scrape_link_async(url_data["url"])
    text = text[:10000]
    gemini_llm_v2 = await get_gemini_llm_v2()
    summary = await gemini_llm_v2.ainvoke(SUMMARY_PROMPT.format(text=text, question=url_data["question"]))
    if summary:
        print("\nSummary content successfuly!\n")
    return f"URL: {url_data['url']}\n\nSummary: {summary}"

SUMMARY_TEMPLATE = generate_webpage_summary_template()
SUMMARY_PROMPT = ChatPromptTemplate.from_template(SUMMARY_TEMPLATE)