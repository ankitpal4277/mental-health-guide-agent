import os
import logging
import google.cloud.logging
from dotenv import load_dotenv

from google.adk import Agent
from google.adk.tools.tool_context import ToolContext
from google.adk.tools.langchain_tool import LangchainTool

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

# --- Setup Logging and Environment ---
cloud_logging_client = google.cloud.logging.Client()
cloud_logging_client.setup_logging()

load_dotenv()
model_name = os.getenv("MODEL")

# --- Tool 1: Save user's mood to state ---
def save_mood_to_state(
    tool_context: ToolContext, mood: str
) -> dict[str, str]:
    """Saves the user's mood to the agent's state."""
    tool_context.state["MOOD"] = mood
    logging.info(f"[State updated] Mood saved: {mood}")
    return {"status": "success"}

# --- Tool 2: Wikipedia for mental health facts ---
wikipedia_tool = LangchainTool(
    tool=WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())
)

# --- The Agent ---
root_agent = Agent(
    name="mental_health_guide",
    model=model_name,
    description="A compassionate mental health check-in agent.",
    instruction="""
    You are a warm, empathetic mental health support guide. 

    When a conversation starts, greet the user warmly and ask for there name. For example:
    "Hello! I'm your Mental Health Check-in Guide. I'm here to listen and support you. 
    How are you feeling today?"
    
    When the user shares how they are feeling:
    1. First, use the 'save_mood_to_state' tool to save their mood.
    2. Then, use the Wikipedia tool to look up one helpful, relevant 
       mental health topic based on what they shared 
       (e.g. "anxiety coping techniques", "mindfulness meditation", 
       "stress management").
    3. Respond with:
       - Acknowledging their feelings with empathy
       - One helpful insight or coping tip from your Wikipedia research
       - A gentle reminder that professional help is always available
    
    Important rules:
    - Never diagnose the user
    - Never say anything discouraging
    - Always be warm, gentle, and non-judgmental
    - If the user mentions self-harm or crisis, immediately 
      provide this message:
      "Please reach out to a crisis helpline immediately. 
       In India, iCall is available at 9152987821."
    """,
    tools=[save_mood_to_state, wikipedia_tool],
)