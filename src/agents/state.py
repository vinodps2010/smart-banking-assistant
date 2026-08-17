from typing import (
    TypedDict,
    Optional,
    Any,
    List,
    Annotated,
)

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):

    # ---------------------------------------------------------
    # User query
    # ---------------------------------------------------------

    query: str

    # Original user query
    original_query: str

    # Rewritten query used for retry
    rewritten_query: Optional[str]

    # ---------------------------------------------------------
    # Routing / Guardrail
    # ---------------------------------------------------------

    route: str

    guardrail: Optional[str]

    # Indicates whether the fast local small-talk
    # check has already been performed.
    fast_small_talk_checked: bool


    # ---------------------------------------------------------
    # Direct response
    # Greeting / small talk
    # ---------------------------------------------------------

    direct_response: Optional[str]

    # ---------------------------------------------------------
    # Retry / Agentic controls
    # ---------------------------------------------------------

    retry_count: int

    max_retries: int

    retrieval_quality: Optional[float]

    retry_required: bool

    # ---------------------------------------------------------
    # RAG
    # ---------------------------------------------------------

    rag_response: dict

    sources: List[Any]

    # ---------------------------------------------------------
    # SQL
    # ---------------------------------------------------------

    sql_response: dict

    # ---------------------------------------------------------
    # Final response
    # ---------------------------------------------------------

    final_response: str

    # ---------------------------------------------------------
    # Conversation memory
    #
    # LangGraph uses add_messages to append new messages
    # to the existing conversation state.
    #
    # PostgreSQL checkpointing persists this state for the
    # current thread/session.
    # ---------------------------------------------------------

    messages: Annotated[
        List[BaseMessage],
        add_messages,
    ]
