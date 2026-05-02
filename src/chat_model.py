from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

# The system prompt defines WHO the chatbot is
SYSTEM_PROMPT = SystemMessage(content="""
You are an expert career coach specialising in tech and AI engineering roles.
You help users identify skill gaps, prepare for interviews, and plan career moves.
Be concise, practical, and encouraging. Ask clarifying questions when needed.
""")

def get_llm():
    """Returns a configured Gemini 2.5 Flash model.

    gemini-2.5-flash: Google's latest fast and capable model (as of 2025-26).
    temperature=0.7: Controls creativity. 0 = robotic/deterministic, 1 = creative/varied.
    """
    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
    )

def chat_with_memory(user_input: str, history: list) -> tuple[str, list]:
    """
    Sends a user message to Gemini and returns the reply + updated history.

    Args:
        user_input: What the user typed
        history: List of previous messages (HumanMessage / AIMessage objects)

    Returns:
        (reply_text, updated_history) — a tuple of the AI's reply and the new history list
    """
    llm = get_llm()

    # Build the full message list: system persona + all past messages + new message
    # This is how the model "remembers" the conversation!
    messages = [SYSTEM_PROMPT] + history + [HumanMessage(content=user_input)]

    # Question: Sending entire message again to llm will use context window???
    # Send to Gemini and get back an AIMessage
    response = llm.invoke(messages)

    # Add this exchange (human + AI) to history for the next turn
    updated_history = history + [
        HumanMessage(content=user_input),
        AIMessage(content=response.content),
    ]

    return response.content, updated_history
