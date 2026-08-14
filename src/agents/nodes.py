"""
LangGraph agent nodes for Smart Banking Assistant.


Nodes:
1. classify_query
2. rag_node
3. rephrase_query_node
4. sql_node
5. merge_node
"""


from src.agents.state import AgentState


from src.services.rag_service import (
    answer_rag_query,
)


from src.services.sql_service import (
    answer_sql_query,
)


from src.agents.query_rewriter import (
    rewrite_query,
)


# ============================================================
# Query Classifier
# ============================================================




def classify_query(
    state: AgentState,
):
    """
    Decide whether query should use:


    - RAG
    - SQL
    - BOTH
    """


    query = state["query"].lower()


    sql_keywords = [
        "account",
        "transaction",
        "balance",
        "amount",
        "statement",
        "withdrawal",
        "deposit",
        "loan outstanding",
        "emi",
        "customer",
    ]


    rag_keywords = [
        "policy",
        "eligibility",
        "criteria",
        "rate",
        "charge",
        "document",
        "requirement",
        "interest",
        "product",
    ]


    if any(keyword in query for keyword in sql_keywords):


        route = "sql"


    elif any(keyword in query for keyword in rag_keywords):


        route = "rag"


    else:


        # Default to RAG
        route = "rag"


    print(f"[agent] Route selected: {route}")


    return {
        "route": route,
        "original_query": state.get(
            "original_query",
            state["query"],
        ),
        "retry_count": state.get(
            "retry_count",
            0,
        ),
        "max_retries": state.get(
            "max_retries",
            1,
        ),
    }




# ============================================================
# RAG Node
# ============================================================




def rag_node(
    state: AgentState,
):
    """
    Execute Hybrid Search + RRF +
    Cohere Reranking RAG pipeline.


    If retrieval quality is poor,
    mark retry_required=True.
    """


    query = state.get("rewritten_query") or state["query"]


    result = answer_rag_query(query)


    retry_count = state.get(
        "retry_count",
        0,
    )


    return {
        "rag_response": result,
        "sources": result.get(
            "sources",
            [],
        ),
        "retrieval_quality": result.get(
            "retrieval_quality",
            0.0,
        ),
        "retry_required": result.get(
            "retry_required",
            False,
        ),
        "retry_count": retry_count,
    }




# ============================================================
# RAG Retry Decision
# ============================================================




def decide_rag_retry(
    state: AgentState,
):
    """
    Decide whether RAG should retry
    with rewritten query.
    """


    retry_required = state.get(
        "retry_required",
        False,
    )


    retry_count = state.get(
        "retry_count",
        0,
    )


    max_retries = state.get(
        "max_retries",
        1,
    )


    if retry_required and retry_count < max_retries:


        return "retry"


    return "finish"




# ============================================================
# Query Rephrase Node
# ============================================================




def rephrase_query_node(
    state: AgentState,
):
    """
    Rewrite weak retrieval query.


    Uses previous retrieved context
    to generate a better search query.
    """


    rag_response = state.get(
        "rag_response",
        {},
    )


    sources = rag_response.get(
        "sources",
        [],
    )


    context_parts = []


    for source in sources[:5]:


        context_parts.append(
            source.get(
                "content",
                "",
            )
        )


    context = "\n\n".join(context_parts)


    rewritten_query = rewrite_query(
        query=state["query"],
        context=context,
    )


    print(
        "[agent] Rewritten query:",
        rewritten_query,
    )


    return {
        "rewritten_query": rewritten_query,
        "retry_count": state.get(
            "retry_count",
            0,
        )
        + 1,
        "retry_required": False,
    }




# ============================================================
# SQL Node
# ============================================================




def sql_node(
    state: AgentState,
):
    """
    Execute SQL based banking queries.
    """


    result = answer_sql_query(state["query"])


    return {
        "sql_response": result,
    }




# ============================================================
# Merge Node
# ============================================================




def merge_node(
    state: AgentState,
):
    """
    Merge RAG / SQL response
    into final answer.
    """


    route = state.get("route")


    if route == "sql":


        sql_response = state.get(
            "sql_response",
            {},
        )


        return {
            "final_response": sql_response.get(
                "answer",
                "No response available.",
            )
        }


    rag_response = state.get(
        "rag_response",
        {},
    )


    return {
        "final_response": rag_response.get(
            "answer",
            "No response available.",
        )
    }
