from typing import TypedDict, Optional, Any, List


class AgentState(TypedDict, total=False):

    # User input
    query: str

    # Classification result
    route: str
    # values:
    # rag
    # sql
    # both

    # RAG output
    rag_response: dict

    # SQL output
    sql_response: dict

    # Final answer
    final_response: str

    # Sources / citations
    sources: List[Any]
