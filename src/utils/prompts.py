def generate_context_and_url_prompt():
  return """Analyze the following input and return only a list in this exact format: ["context_of_the_request", "url_from_input"]

      Rules:
      1. If the input contains both text and a URL, extract the context and the URL, regardless of their order.
      2. If the input is ONLY a URL, return it as the only element in the list.
      3. Do NOT include any extra text, explanation, or formatting of any kind.
      4. Trim all leading/trailing whitespace from both context and URL.

      Example 1:
      User input: Summarize this YouTube video in 3 lines: https://youtu.be/Huy-Gn4RtGQ?si=TCBo-fVBxNWXdDh6
      Expected output: ["Summarize this YouTube video in 3 lines", "https://youtu.be/Huy-Gn4RtGQ?si=TCBo-fVBxNWXdDh6"]

      Example 2:
      User input:  https://youtu.be/Huy-Gn4RtGQ?si=TCBo-fVBxNWXdDh6
      Expected output: ["https://youtu.be/Huy-Gn4RtGQ?si=TCBo-fVBxNWXdDh6"]

      Example 3:
      User input:  https://youtu.be/dQw4w9WgXcQ?si=123456789 resume the YouTube video in 3 lines
      Expected output: ["resume the YouTube video in 3 lines", "https://youtu.be/dQw4w9WgXcQ?si=123456789"]

      Now process this input:
      User input: {input}"""

def generate_youtube_transcribe_prompt():
  # return """Transcribe the audio from this video, giving timestamps for salient events in the video. Also provide visual descriptions."""
  return '''Transcribe the audio from this video, giving timestamps for salient events in the video.
            Also provide visual descriptions. Respond in the same language as the request.'''

def generate_webpage_summary_template():
	return """{text}

	-------------------

	Using the above text, answer in short the following question:

	> {question}

	-------------------
	If the question cannot be answered using the text, simply summarize the text. 
	Include all factual information, numbers, stats, etc.

	Respond in the same language the question is written in.
	"""
#   return """{text}

#   -------------------

#   Using the above text, answer in short the following question:

#   > {question}

#   -------------------
#   if the question cannot be answered using the text, simply summarize the text. Include all factual information, numbers, stats, etc.
# """

def generate_search_queries_prompt():
    return '''Generate exactly 3 Google search queries to search online that form an objective opinion from the following task: "{question}".
            Include specific details such as locations, names, etc.
            Return only a list of search queries strictly in the following format: ["query 1", "query 2", "query 3"].
            Respond in the same language as the task.'''

  # return '''Generate exactly 3 Google search queries related to "{question}".
  #   Include specific details such as locations, names, etc.
  #   Example format: ["Query 1", "Query 2", "Query 3"]
  #   Return only a valid JSON array of strings with no extra text.'''

#   return 'Generate exactly 3 google search queries to search online that form an objective opinion from the following task: "{question}"' \
#           'Also include in the queries specified task details such as locations, names, etc.\n' \
#           'You must respond with a list of search queries strictly in the following format: ["query 1", "query 2", "query 3"].'

def generate_research_report_prompt():
    return (
        "Information:\n"
        "{context}\n\n"
        "Using the above information, answer the following query or task: \"{question}\" in a detailed report.\n"
        "The report should be:\n"
        "- Well structured and informative\n"
        "- In depth and comprehensive (minimum of 1200 words)\n"
        "- Written in Markdown syntax\n"
        "- Unbiased and journalistic in tone\n"
        "- Include relevant facts, numbers, and data\n"
        "- Contain hyperlinks to source URLs at the end as references, without duplicates (use: [source name](url))\n"
        "- Cite search results inline only if they are the most relevant\n"
        "- Include inline hyperlinks wherever URLs are mentioned in the text\n\n"
        "Important:\n"
        "- You MUST derive a valid and concrete opinion based on the information\n"
        "- Do NOT conclude with vague generalities\n"
        "- Respond in the same language as the user's request\n"
        "- Please do your best, this is very important to my career."
    )
	
#   return 'Information: """{context}"""\n\n' \
#           'Using the above information, answer the following' \
#           ' query or task: "{question}" in a detailed report --' \
#            " The report should focus on the answer to the query, should be well structured, informative," \
#           " in depth and comprehensive, with facts and numbers if available and a minimum of 1200 words.\n" \
#           "You should strive to write the report as long as you can using all relevant and necessary information provided.\n" \
#           "You must write the report with markdown syntax.\n " \
#           "Use an unbiased and journalistic tone. \n" \
#           "You MUST determine your own concrete and valid opinion based on the given information. Do NOT deter to general and meaningless conclusions.\n" \
#           "You MUST write all used source urls at the end of the report as references, and make sure to not add duplicated sources, but only one reference for each.\n" \
#           "Every url should be hyperlinked: [url website](url)"\
#           """
#             Additionally, you MUST include hyperlinks to the relevant URLs wherever they are referenced in the report : 
        
#             eg:    
#                 # Report Header
                
#                 This is a sample text. ([url website](url))
#             """\
#           "Cite search results using inline notations. Only cite the most \
#           relevant results that answer the query accurately. Place these citations at the end \
#           of the sentence or paragraph that reference them.\n"\
#           "Please do your best, this is very important to my career."

def generate_critical_thinker_prompt():
    return (
        "You are an AI critical thinker research assistant. Your sole purpose is to write well written, "
        "critically acclaimed, objective and structured reports on given text."
    )