from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

class SkillGapAnalysis(BaseModel):
    """Structured skill gap analysis result.

    Using Pydantic BaseModel ensures the AI's JSON output is validated
    and converted to a proper Python object with type checking.
    """
    target_role: str = Field(description="The job role being analysed")
    skills_you_have: list[str] = Field(description="Skills the user already has")
    skills_to_learn: list[str] = Field(description="Skills the user needs to develop")
    recommended_courses: list[str] = Field(description="Free courses to fill the gaps")
    estimated_months: int = Field(description="Estimated months to become job-ready")

def build_skill_gap_chain():
    """
    Returns an LCEL chain that forces structured JSON output.

    with_structured_output() is the modern LangChain approach —
    it instructs the model to return JSON matching the Pydantic schema.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    # with_structured_output wraps the LLM to enforce the Pydantic schema
    structured_llm = llm.with_structured_output(SkillGapAnalysis)

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a career coach. Analyse skill gaps for the given role. "
            "Be specific and practical. Recommend free resources only."
        )),
        ("human", (
            "I want to become a {target_role}. "
            "My current skills are: {current_skills}"
        )),
    ])

    # LCEL chain: prompt | structured_llm
    # The output will be a SkillGapAnalysis object automatically!
    chain = prompt | structured_llm

    return chain

def analyse_skill_gap(target_role: str, current_skills: str) -> SkillGapAnalysis:
    """
    Runs the skill gap analysis chain and returns a structured result.

    Args:
        target_role: The job title to aim for (e.g., "AI Engineer")
        current_skills: Comma-separated list of current skills

    Returns:
        A SkillGapAnalysis Pydantic object with structured data
    """
    chain = build_skill_gap_chain()
    result = chain.invoke({
        "target_role": target_role,
        "current_skills": current_skills,
    })

    return result

def simple_advice_chain(question: str) -> str:
    """
    A simple LCEL chain using StrOutputParser — returns plain text.

    This demonstrates the simplest possible LCEL pattern:
    prompt | llm | StrOutputParser()
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.7)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a career coach. Give a short, practical answer in 2-3 sentences."),
        ("human", "{question}"),
    ])

    # StrOutputParser extracts just the text string from the AIMessage
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"question": question})