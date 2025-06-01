# Step 1: Setup API Keys (use .env or set them manually)
from dotenv import load_dotenv
import os

load_dotenv()

# Add validation for required API keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file")

TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step 2: Setup LLMs & Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any

# These are just default LLMs — you can override them in the function
openai_llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
groq_llm = ChatGroq(model="llama3-70b-8192", api_key=GROQ_API_KEY)

search_tool = TavilySearchResults(max_results=2) if TAVILY_API_KEY else None

# Step 3: Create Response Function
from langgraph.prebuilt import create_react_agent

default_system_prompt = "Act as an AI chatbot who is smart and friendly"


def format_message_for_groq(message: str, role: str = "user") -> Dict[str, str]:
    return {
        "role": role,
        "content": message
    }


def get_response_from_ai_agent(llm_id: str, query: str, allow_search: bool, system_prompt: str, provider: str) -> str:
    if provider == "Groq":
        if not GROQ_API_KEY:
            raise ValueError("Groq API key is required but not found in environment variables")
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query)
        ]

        response = llm.invoke(messages)
        return response.content if hasattr(response, 'content') else str(response)

    elif provider == "OpenAI":
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key is required but not found in environment variables")
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)
    else:
        raise ValueError("Provider must be either 'Groq' or 'OpenAI'")

    tools = [search_tool] if (allow_search and search_tool) else []

    try:
        agent = create_react_agent(
            model=llm,
            tools=tools,
        )

        state = {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
        }

        response = agent.invoke(state)
        messages = response.get("messages", [])
        ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
        return ai_messages[-1] if ai_messages else "No response from agent."

    except Exception as e:
        return f"Error: {str(e)}"
