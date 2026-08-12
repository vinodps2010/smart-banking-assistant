from langgraph.graph import StateGraph, END


from src.agents.state import AgentState


from src.agents.nodes import classify_query, rag_node, sql_node, merge_node

workflow = StateGraph(AgentState)


workflow.add_node("classifier", classify_query)


workflow.add_node("rag", rag_node)


workflow.add_node("sql", sql_node)


workflow.add_node("merge", merge_node)


workflow.set_entry_point("classifier")


workflow.add_conditional_edges(
    "classifier",
    lambda state: state["route"],
    {"rag": "rag", "sql": "sql", "both": "rag"},
)


workflow.add_edge("rag", "merge")


workflow.add_edge("sql", "merge")


workflow.add_edge("merge", END)


agent_graph = workflow.compile()
