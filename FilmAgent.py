import asyncio
from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool
from google.genai import types
from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
import json
from dotenv import load_dotenv
import requests

import os
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")

def web_search(query:str) -> str:
    """Search the web using Google Custom Search API and return the first snippet."""

    url="https://www.googleapis.com/customsearch/v1"

    params={
        "key":os.getenv("CUSTOMSEARCH_API_KEY"),
        "cx":os.getenv("CSE_ID"),
        "q":query

    }
    
    response = requests.get(url, params=params)
    data = response.json()

    # Extract the first search result’s snippet
    if "items" in data and data["items"]:
        snippet = data["items"][0].get("snippet", "No snippet found.")
        return snippet
    else:
        return "No results found."




APP_NAME="film_app"
USER_ID="film_1221"
MODEL_NAME="gemini-2.0-flash"
SESSION_ID_TOOL_AGENT="session_tool_agent_xyz"

search_tool=FunctionTool(web_search)

film_agent=LlmAgent(
    model=MODEL_NAME,
    name="film_agent",
    description="Summarizes and reviews films.",
    instruction="""You are an agent that sumarizes and reviews films
    When a user inputs a film name:
    1.Search the web using tool for relevant information.after_agent_callback.
    2.Summarize it into a clear and attractive format.
    3.Gather reviews about the film.
    4.Output the summary.
    5.Output top two good and bad reviews.
    6.Respond clearly to user outputting only summary and reviews.
    7.Check for Movie within Indian Cinema only
    8.Strictly do not answer any other queries apart from movie review
    9.If asked anything other than movie review reply with a polite response
    10.Before each answer do a fact check and reply only isf movie found else reject politely
    Example Query:Review movie Life of Pi
    Output:Pi Patel finds a way to survive in a lifeboat that is adrift in the middle of nowhere. His fight against the odds is heightened by the company of a hyena and a tiger in his vessel.
    86% Rotten Tomatoes
7.9/10 IMDb
4/5 Times of India
    """,
    tools=[search_tool],
    generate_content_config=types.GenerateContentConfig(
        temperature=0.4,
        max_output_tokens=200
    )
)

session_service=InMemorySessionService()



film_runner=Runner(
    agent=film_agent,
    app_name=APP_NAME,
    session_service=session_service
)

async def call_agent_and_print(
    runner_instance: Runner,
    agent_instance: LlmAgent,
    session_id: str,
    query_json: str
):
    """Sends a query to the specified agent/runner and prints results."""
    print(f"\n>>> Calling Agent: '{agent_instance.name}' | Query: {query_json}")

    user_content = types.Content(role='user', parts=[types.Part(text=query_json)])

    final_response_content = "No final response received."
    async for event in runner_instance.run_async(user_id=USER_ID, session_id=session_id, new_message=user_content):
        # print(f"Event: {event.type}, Author: {event.author}") # Uncomment for detailed logging
        if event.is_final_response() and event.content and event.content.parts:
            final_response_content = event.content.parts[0].text

    print(f"<<< Agent '{agent_instance.name}' Response: {final_response_content}")

    current_session = await session_service.get_session(app_name=APP_NAME,
                                                  user_id=USER_ID,
                                                  session_id=session_id)
    stored_output = current_session.state.get(agent_instance.output_key)

    print(f"--- Session State ['{agent_instance.output_key}']: ", end="")
    try:
        parsed_output = json.loads(stored_output)
        print(json.dumps(parsed_output, indent=2))
    except (json.JSONDecodeError, TypeError):
        print(stored_output)
    print("-" * 30)


async def main():
    await session_service.create_session(
    app_name=APP_NAME,
    user_id=USER_ID, 
    session_id=SESSION_ID_TOOL_AGENT
    )
    print("----Agents----")
    query=input("Enter the movie you to hear review  :")
    await call_agent_and_print(film_runner,film_agent,SESSION_ID_TOOL_AGENT,query)

if __name__=="__main__":
    while(True):
        asyncio.run(main())

