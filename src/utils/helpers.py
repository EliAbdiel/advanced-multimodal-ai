import json
from .llm_setup import get_gemini_llm

async def json_loads(text):
	"""Safely loads JSON from a string, attempting to fix common issues."""
	try:
		start = text.find('[')
		end = text.find(']')
		if start != -1 and end != -1:
			json_str = text[start:end+1]
			return json.loads(json_str)
		return json.loads(text)
	except json.JSONDecodeError as e:
		print(f"Initial JSON parse failed: {e}")
		try:
			gemini_llm = await get_gemini_llm()
			fixed_json = await gemini_llm.ainvoke(f'Fix this JSON and return a list of search queries strictly in the following format ["query 1", "query 2", "query 3"]: {text}')
			return json.loads(fixed_json.content.strip())
		except Exception as fix_error:
			print(f"Failed to fix JSON: {fix_error}")
			return []
    
async def safe_json_loads(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"Initial JSON parse failed: {e}")
        try:
            gemini_llm = await get_gemini_llm()
            fixed_json = await gemini_llm.ainvoke(f'Fix this JSON and return only the corrected format ["context_of_the_request", "url_from_input"]: {text}')
            return json.loads(fixed_json.content.strip())
        except Exception as fix_error:
            print(f"Failed to fix JSON: {fix_error}")
            return []