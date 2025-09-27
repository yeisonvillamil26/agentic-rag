from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableSequence
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class GradeHallucination(BaseModel):
    """Binary score for hallucination present in generation answer"""

    binary_score: str = Field(
        description="Binary score for hallucination present in generation answer"
    )


structured_llm_grader = llm.with_structured_output(GradeHallucination)

system = """You are a grader assessing whether a LLM generation is ground truth or not supported of a set of retrieved 
        facts.
        Give a binary score 'yes' or 'no' score to indicate whether the generation is ground truth or not supported of a 
        set of retrieved facts"""

hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        (
            "human",
            "Set of facts: \n\n {documents} \n\n LLM generation: \n\n {generation}",
        ),
    ]
)

hallucination_grader: RunnableSequence = hallucination_prompt | structured_llm_grader
