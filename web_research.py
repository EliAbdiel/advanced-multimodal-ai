import os
# import io
# import markdown
# import time
# import random
# import itertools
# import requests
# from weasyprint import HTML
import aiohttp
import asyncio
import chainlit as cl

from ddgs import DDGS
from dotenv import load_dotenv
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from bs4 import BeautifulSoup
from search_duckduckgo_queries import duckduckgo_search

load_dotenv()

RESULTS_PER_QUESTION = 1

ddgs = DuckDuckGoSearchAPIWrapper(time='y')

tavily_search = TavilySearchAPIWrapper(tavily_api_key=os.environ["TAVILY_API_KEY"])

# def scrape_link(url):
#   try:
#     response = requests.get(url)
    
#     if response.status_code == 200:
#       # Parse the content of the page using BeautifulSoup
#       soup = BeautifulSoup(response.text, 'html.parser')
        
#       # Extract all text from the webpage
#       text = soup.get_text(separator=' ', strip=True)

#       return text
#     else:
#       return f"Failed to scrape the webpage. Status code: {response.status_code}"
#   except Exception as e:
#     print(f"An error occurred: {e}")
#     return f"Failed to scrape the webpage. Error: {e}"

async def scrape_link_async(url):
  """
  Asynchronously scrape a webpage and extract its text content.
  """
  try:
    async with aiohttp.ClientSession() as session:
      async with session.get(url) as response:
        if response.status == 200:
          # Parse the content of the page using BeautifulSoup
          html = await response.text()
          soup = BeautifulSoup(html, 'html.parser')
            
          # Extract all text from the webpage
          text = soup.get_text(separator=' ', strip=True)
          if text:
            print("\nText extracted successfuly!\n")

          return text
        else:
          return f"Failed to scrape the webpage. Status code: {response.status}"
  except Exception as e:
    print(f"An error occurred while I scrape the webpage: {e}")
    return f"Failed to scrape the webpage. Error: {e}"

def flatten_list_of_list(list_of_list):
  """
  Flatten a list of lists into a single list.
  """
  content = []
  for sublist in list_of_list:
    content.append("\n\n".join(sublist))
  return "\n\n".join(content)

async def web_search(query: str, num_results: int = RESULTS_PER_QUESTION):
  """
  Search the web using DuckDuckGo and return a list of URLs.
  """
  try:
    results = await cl.make_async(DDGS().text)(keywords=query, max_results=num_results)
    # print(f"\nresult is a list of dicts: \n{results}\n")
    urls = [r["href"] for r in results]
    print(f"\nURLs found from ddgs: {urls}\n")
    return urls
  except Exception as e:
    print(f"An error occurred while I searched the query: {e}")
    return []

async def web_search_with_tavily(query: str, num_results: int = RESULTS_PER_QUESTION):
  """
  Asynchronously search the web using Tavily and return a list of URLs.
  """
  try:
    result = await tavily_search.results_async(query=query, max_results=num_results)
    # print(f"\nresult is a list of dicts: \n{result}\n")
    urls = [item["url"] for item in result if "url" in item]
    print(f"\nURLs found from tavily: {urls}\n")
    return urls
  except Exception as e:
    print(f"An error occurred while I searched the query: {e}")
    return []

async def web_search_async(query: str, num_results: int = RESULTS_PER_QUESTION):
  """
  Asynchronously search the web using DuckDuckGo and return a list of URLs.
  """
  try:
    queries = []
    queries.append(query)
    results = await duckduckgo_search(search_queries=queries, num_results=num_results)
    print(f"\nduckduckgo search results: {results}\n")
    return results
  except Exception as e:
    print(f"An error occurred while I searched the query: {e}")
    return []

# async def web_search_async(query: str, num_results: int = RESULTS_PER_QUESTION, max_retries: int = 3, delay: float = 2.0):
#   for attempt in range(max_retries):
#     try:
#       # Add a delay between attempts
#       if attempt > 0:
#         await asyncio.sleep(delay * attempt)
      
#       # loop = asyncio.get_event_loop()
#       # results = await loop.run_in_executor(None, lambda: ddgs.results(query, num_results))
#       results = await cl.make_async(ddgs.results)(query, num_results)
#       return [r["link"] for r in results]
#     except Exception as e:
#       if attempt == max_retries - 1:
#         print(f"Error after {max_retries} attempts: {e}")
#         return []
#       print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
