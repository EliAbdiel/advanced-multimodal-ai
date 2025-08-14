import aiohttp
import chainlit as cl

from src.utils.config import TAVILY_KEY
from ddgs import DDGS
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper
from bs4 import BeautifulSoup

RESULTS_PER_QUESTION = 1

tavily_search = TavilySearchAPIWrapper(tavily_api_key=TAVILY_KEY)

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