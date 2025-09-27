from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from graph.const import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.chains.answer_grader import answer_grader
from graph.chains.hallucination_grader import hallucination_grader
from graph.state import GraphState


def decide_to_generate(state):
    print("--ASSESS GRADE DOCUMENTS--")
    if state["web_search"]:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        return WEB_SEARCH
    else:
        print("---GRADE: DOCUMENT RELEVANT---")
        return GENERATE


def grade_generation_grounded_in_documents_and_question(state: GraphState) -> str:
    print("--CHECK HALLUCINATIONS")
    question = state["question"]
    generation = state["generation"]
    documents = state["documents"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )

    if hallucination_grade := score.binary_score:
        print("--DECISION: GENERATION IS GROUNDED IN DOCUMENTS")
        print("--GRADE GENERATION Vs QUESTION--")
        score = answer_grader.invoke({"question": question, "generation": generation})
        if answer_grade := score.binary_score:
            print("--DECISION: GENERATION ADDRESSES QUESTION--")
            return "useful"
        else:
            print("--DECISION: GENERATION DOES NOT ADDRESS QUESTION--")
            return "not useful"
    else:
        print("--DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS")
        return "not supported"


# Initialize workflow
workflow = StateGraph(GraphState)

# Nodes
workflow.add_node(RETRIEVE, retrieve)
workflow.add_node(GRADE_DOCUMENTS, grade_documents)
workflow.add_node(GENERATE, generate)
workflow.add_node(WEB_SEARCH, web_search)

# Initial point
workflow.set_entry_point(RETRIEVE)

# Edge
workflow.add_edge(RETRIEVE, GRADE_DOCUMENTS)

# Conditional edge
workflow.add_conditional_edges(
    GRADE_DOCUMENTS,
    decide_to_generate,
    {WEB_SEARCH: WEB_SEARCH, GENERATE: GENERATE},
)

# Conditional grader
workflow.add_conditional_edges(
    GENERATE,
    grade_generation_grounded_in_documents_and_question,
    {"useful": END, "not useful": WEB_SEARCH, "not supported": GENERATE},
)

# Generation
workflow.add_edge(WEB_SEARCH, GENERATE)
# Workflow end
workflow.add_edge(GENERATE, END)

# Compile
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph_v2.png")
