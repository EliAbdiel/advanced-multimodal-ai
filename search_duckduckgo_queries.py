import chainlit as cl
import asyncio
import random
import time
import httpx

from ddgs import DDGS
from markdownify import markdownify
from typing import List, Optional, Dict, Any, Union, Literal, Annotated, cast


async def duckduckgo_search(search_queries: List[str], num_results: int):
    """Perform searches using DuckDuckGo with retry logic to handle rate limits
    
    Args:
        search_queries (List[str]): List of search queries to process
        
    Returns:
        str: A formatted string of search results
    """
    
    async def process_single_query(query):
        # Execute synchronous search in the event loop's thread pool
        loop = asyncio.get_event_loop()
        
        def perform_search():
            max_retries = 3
            retry_count = 0
            backoff_factor = 2.0
            last_exception = None
            
            while retry_count <= max_retries:
                try:
                    results = []
                    with DDGS() as ddgs:
                        # Change query slightly and add delay between retries
                        if retry_count > 0:
                            # Random delay with exponential backoff
                            delay = backoff_factor ** retry_count + random.random()
                            print(f"Retry {retry_count}/{max_retries} for query '{query}' after {delay:.2f}s delay")
                            time.sleep(delay)
                            
                            # Add a random element to the query to bypass caching/rate limits
                            modifiers = ['about', 'info', 'guide', 'overview', 'details', 'explained']
                            modified_query = f"{query} {random.choice(modifiers)}"
                        else:
                            modified_query = query
                        
                        # Execute search
                        ddg_results = list(ddgs.text(keywords=query, max_results=num_results, timelimit='y'))
                        
                        # Format results
                        for i, result in enumerate(ddg_results):
                            results.append({
                                'title': result.get('title', ''),
                                'url': result.get('href', ''),
                                'content': result.get('body', ''),
                                'score': 1.0 - (i * 0.1),  # Simple scoring mechanism
                                'raw_content': result.get('body', '')
                            })
                        
                        # Return successful results
                        return {
                            'query': query,
                            'follow_up_questions': None,
                            'answer': None,
                            'images': [],
                            'results': results
                        }
                except Exception as e:
                    # Store the exception and retry
                    last_exception = e
                    retry_count += 1
                    print(f"DuckDuckGo search error: {str(e)}. Retrying {retry_count}/{max_retries}")
                    
                    # If not a rate limit error, don't retry
                    if "Ratelimit" not in str(e) and retry_count >= 1:
                        print(f"Non-rate limit error, stopping retries: {str(e)}")
                        break
            
            # If we reach here, all retries failed
            print(f"All retries failed for query '{query}': {str(last_exception)}")
            # Return empty results but with query info preserved
            return {
                'query': query,
                'follow_up_questions': None,
                'answer': None,
                'images': [],
                'results': [],
                'error': str(last_exception)
            }
            
        return await loop.run_in_executor(None, perform_search)

    # Process queries with delay between them to reduce rate limiting
    search_docs = []
    urls = []
    titles = []
    for i, query in enumerate(search_queries):
        # Add delay between queries (except first one)
        if i > 0:
            delay = 2.0 + random.random() * 2.0  # Random delay 2-4 seconds
            await asyncio.sleep(delay)
        
        # Process the query
        result = await process_single_query(query)
        search_docs.append(result)
        
        # Safely extract URLs and titles from results, handling empty result cases
        if result['results'] and len(result['results']) > 0:
            for res in result['results']:
                if 'url' in res and 'title' in res:
                    urls.append(res['url'])
                    titles.append(res['title'])
    
    # If we got any valid URLs, scrape the pages
    if urls:
        # return await scrape_pages(titles, urls)
        return urls
    else:
        # return "No valid search results found. Please try different search queries or use a different search API."
        return []


async def agent_results_text(user_message: str) -> list[dict[str, str, str]]:
    """
    Asynchronously retrieves text search results from DuckDuckGo based on user input.

    Args:
        user_message (str): The query string entered by the user.

    Returns:
        list[dict]: A list of search result dictionaries containing information like title, link, and description.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(keywords=user_message, max_results=5))
    return results

async def text(
    keywords: str,
    max_results: int | None = None,
    ) -> list[dict[str, str, str]]:
    """
    Performs a text search on DuckDuckGo with specified query parameters.

    Args:
        keywords (str): The search keywords.
        max_results (int | None): Maximum number of results to retrieve. If None, defaults to the first response only.

    Returns:
        list[dict]: A list of dictionaries containing the search results.
    """
    with DDGS() as ddgs:
        results = list(ddgs.text(keywords=keywords, max_results=5))
    return results

