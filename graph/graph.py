from dotenv import load_dotenv

load_dotenv()

from langgraph.graph import StateGraph, END
from graph.const import RETRIEVE, GRADE_DOCUMENTS, GENERATE, WEB_SEARCH
from graph.nodes import retrieve, grade_documents, generate, web_search
from graph.state import GraphState


def decide_to_generate(state):
    print("--ASSESS GRADE DOCUMENTS--")
    if state["web_search"]:
        print("---GRADE: DOCUMENT NOT RELEVANT---")
        return WEB_SEARCH
    else:
        print("---GRADE: DOCUMENT RELEVANT---")
        return GENERATE


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

# Generation
workflow.add_edge(WEB_SEARCH, GENERATE)
# Workflow end
workflow.add_edge(GENERATE, END)

# Compile
app = workflow.compile()
app.get_graph().draw_mermaid_png(output_file_path="graph.png")
