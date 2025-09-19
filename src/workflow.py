import chainlit as cl
import json

from typing import Dict
from langchain_core.messages import HumanMessage
from src.services.file_processing import handle_file_processing
from src.core.graph_builder import build_graph
from src.services.pdf_processing import content_as_pdf


async def run_agent_workflow(user_message: cl.Message):
    """Run the agent workflow based on the user message"""
    async with build_graph() as app:

        # Prepare the config with thread_id
        config = {
            "configurable": {
                "thread_id": cl.context.session.thread_id
            }
        }

        if user_message.command:
            pass

        if user_message.elements:
            result = await handle_file_processing(user_message)
            print(f"\n{result}\n")
            await cl.Message(content=result, elements=[]).send()

        else:
            # Get the last user message
            user_msg = [HumanMessage(content=user_message.content)]

            result = await app.ainvoke({"messages": user_msg}, config=config)

            print(f"\n{result}\n")

            if result["messages"][-1].name == "generate_image":
                image_element = cl.Image(name="Generated Image", path=result["messages"][-1].content)
                await cl.Message(content="Here's the generated image!", elements=[image_element]).send()

            elif result["messages"][-1].name == "deep_research_report":
                search_results = result["messages"][-1].content
                
                if len(search_results) > 100:
                    pdf_path = await content_as_pdf(content=search_results)
                    pdf_element = cl.Pdf(name="Research Report", path=str(pdf_path))
                    await cl.Message(content=search_results, elements=[pdf_element]).send()
                
                else:
                    await cl.Message(content=search_results, elements=[]).send()
            
            elif result["messages"][-1].name == "youtube_transcribe":
                content = json.loads(result["messages"][-1].content)
                youtube = cl.Video(name="YouTube Video", url=content[1])
                await cl.Message(content=content[0], elements=[youtube]).send()

            # elif result["messages"][-1].name == "generate_video":
            #     video_path = result["messages"][-1].content
            #     video = cl.Video(name="Generated Video", path=str(video_path))
            #     await cl.Message(content="Here's the generated video!", elements=[video]).send()
            
            else:
                await cl.Message(content=result["messages"][-1].content, elements=[]).send()

