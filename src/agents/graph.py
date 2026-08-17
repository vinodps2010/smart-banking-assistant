"""
LangGraph workflow for Smart Banking Assistant.

Flow:

                    START
                      |
                      v
               small_talks
                /        \
               /          \
      fast match          continue
          |                  |
          v                  v
         END            classifier
                         /   |   |   \
                        /    |   |    \
                       v     v   v     v
                     rag   sql both  small_talks
                      |     |    |       |
                    retry  merge merge   END
                      |
                   rephrase
                      |
                      v
                     rag
"""

from langgraph.graph import (
    StateGraph,
    END,
)

from src.agents.checkpointer import (
    create_checkpointer,
)

from src.agents.state import AgentState

from src.agents.nodes import (
    small_talks_response_node,
    classify_query,
    rag_node,
    sql_node,
    both_node,
    merge_node,
    rephrase_query_node,
    decide_rag_retry,
)

from src.common.logger import logger

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
    "small_talks",
    small_talks_response_node,
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
    "both",
    both_node,
)

workflow.add_node(
    "merge",
    merge_node,
)


# ============================================================
# Entry Point
# ============================================================

workflow.set_entry_point("small_talks")


# ============================================================
# Fast Small-Talk / Classifier Routing
# ============================================================

workflow.add_conditional_edges(
    "small_talks",
    lambda state: state["route"],
    {
        # Fast local small-talk match
        "small_talks": END,
        # No fast match -> LLM classifier
        "continue": "classifier",
    },
)


# ============================================================
# Classifier Routing
# ============================================================

workflow.add_conditional_edges(
    "classifier",
    lambda state: state["route"],
    {
        "rag": "rag",
        "sql": "sql",
        "both": "both",
        # LLM classified conversational/unrelated request
        "small_talks": "small_talks",
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
# BOTH Path
# ============================================================

workflow.add_edge(
    "both",
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

    checkpointer_context.__exit__(
        None,
        None,
        None,
    )


# command for generating graph
# uv run python -m src.agents.graph
