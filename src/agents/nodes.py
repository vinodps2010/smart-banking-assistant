from src.agents.state import AgentState

from src.services.rag_service import answer_rag_query

# from src.sql.sql_engine import process_natural_language_query
from src.services.sql_service import answer_sql_query


def classify_query(state: AgentState):

    query = state["query"].lower()

    sql_keywords = [
        "transaction",
        "transactions",
        "account",
        "balance",
        "statement",
        "withdraw",
        "deposit",
        "amount",
    ]

    rag_keywords = [
        "home loan",
        "loan eligibility",
        "interest rate",
        "credit card",
        "fixed deposit",
        "charges",
        "product",
    ]

    has_sql = any(word in query for word in sql_keywords)

    has_rag = any(word in query for word in rag_keywords)

    if has_sql and has_rag:
        route = "both"

    elif has_sql:
        route = "sql"

    else:
        route = "rag"

    print(f"[agent] Route selected: {route}")

    return {"route": route}


def rag_node(state: AgentState):

    result = answer_rag_query(state["query"])

    return {"rag_response": result, "sources": result["sources"]}


def sql_node(state: AgentState):

    result = answer_sql_query(state["query"])

    return {"sql_response": result}


def route_handler(state):
    route = state["route"]
    return state


def merge_node(state: AgentState):
    route = state["route"]
    if route == "rag":
        answer = state["rag_response"]["answer"]
    elif route == "sql":
        # answer = str(state["sql_response"])
        answer = state["sql_response"]["answer"]
    else:
        answer = (
            "RAG Result:\n\n"
            + str(state.get("rag_response"))
            + "\n\nSQL Result:\n\n"
            + str(state.get("sql_response"))
        )

    return {"final_response": answer}
