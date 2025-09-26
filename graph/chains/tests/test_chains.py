from dotenv import load_dotenv
from pprint import pprint

from graph.chains.generation import generation_chain

load_dotenv()

from graph.chains.retrieval_grader import retrieval_grader, GradeDocuments
from ingestion import retriever


def test_foo() -> None:
    assert 1 == 1


def test_retrieval_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": question}
    )
    assert res.binary_score == "yes"


def test_retrieval_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[0].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"document": doc_txt, "question": "how to make pizza"}
    )
    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)
