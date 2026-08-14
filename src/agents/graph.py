"""
LangGraph workflow for Smart Banking Assistant.


Flow:


                    User Query
                         |
                         v
                  Query Classifier
                         |
             +-----------+-----------+
             |                       |
             v                       v
           RAG                     SQL
             |                       |
             v                       v
     Hybrid Search + RRF        PostgreSQL
             |
             v
      Cohere Reranking
             |
             v
      Relevance Check
             |
        +----+----+
        |         |
      Good       Poor
        |         |
        v         v
      Merge    Rephrase Query
                  |
                  v
              Retry RAG
                  |
                  v
                Merge


"""


from langgraph.graph import (
    StateGraph,
    END,
)


# from src.agents.checkpointer import get_checkpointer
from src.agents.checkpointer import create_checkpointer


from src.agents.state import AgentState




from src.agents.nodes import (
    classify_query,
    rag_node,
    sql_node,
    merge_node,
    rephrase_query_node,
    decide_rag_retry,
)


# ============================================================
# Create Graph
# ============================================================




workflow = StateGraph(AgentState)




# ============================================================
# Register Nodes
# ============================================================




workflow.add_node(
    "classifier",
    classify_query,
)




workflow.add_node(
    "rag",
    rag_node,
)




workflow.add_node(
    "rephrase",
    rephrase_query_node,
)




workflow.add_node(
    "sql",
    sql_node,
)




workflow.add_node(
    "merge",
    merge_node,
)




# ============================================================
# Entry Point
# ============================================================




workflow.set_entry_point("classifier")




# ============================================================
# Classifier Routing
# ============================================================




workflow.add_conditional_edges(
    "classifier",
    lambda state: state["route"],
    {
        "rag": "rag",
        "sql": "sql",
        # keep existing BOTH handling
        # future enhancement:
        # parallel RAG + SQL execution
        "both": "rag",
    },
)




# ============================================================
# RAG Retry Decision
# ============================================================




workflow.add_conditional_edges(
    "rag",
    decide_rag_retry,
    {
        # Poor retrieval
        "retry": "rephrase",
        # Good retrieval
        "finish": "merge",
    },
)




# ============================================================
# Retry Loop
# ============================================================




workflow.add_edge(
    "rephrase",
    "rag",
)




# ============================================================
# SQL Path
# ============================================================




workflow.add_edge(
    "sql",
    "merge",
)




# ============================================================
# Final Response
# ============================================================




workflow.add_edge(
    "merge",
    END,
)




# ============================================================
# Checkpointer
# ============================================================


checkpointer, checkpointer_context = create_checkpointer()




# ============================================================
# Compile Graph
# ============================================================


agent_graph = workflow.compile(checkpointer=checkpointer)




# ============================================================
# Graph visualization
# ============================================================


if __name__ == "__main__":


    graph_image = agent_graph.get_graph().draw_mermaid_png()


    with open(
        "src/agents/graph.png",
        "wb",
    ) as f:
        f.write(graph_image)


    print("Graph image generated: src/agents/graph.png")
