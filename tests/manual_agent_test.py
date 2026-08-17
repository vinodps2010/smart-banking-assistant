from src.agents.graph import agent_graph

queries = [
    # "What is the eligibility criteria for home loan?",
    # "Show transactions for account 1345367 where amount is greater than 50000",
    # "what do i need to get it"
    "Delete all customer accounts",
    "Update customer mobile number",
    "Show customer details for account 1345367",
    "Drop all tables"
]


for query in queries:

    print("\n===================")

    print("QUERY:", query)

    # result = agent_graph.invoke({"query": query})

    config = {"configurable": {"thread_id": "manual-test-001"}}

    result = agent_graph.invoke(
        {
            "query": query,
            "original_query": query,
            "retry_count": 0,
            "max_retries": 1,
            "rewritten_query": None,
        },
        config=config,
    )
    print("ROUTE:", result["route"])

    print("ANSWER:")

    print(result["final_response"])


# uv run python -m tests.manual_agent_test
