from src.agents.graph import agent_graph

queries = [
    "What is the eligibility criteria for home loan?",
    "Show transactions for account 1345367 where amount is greater than 50000",
]


for query in queries:

    print("\n===================")

    print("QUERY:", query)

    result = agent_graph.invoke({"query": query})

    print("ROUTE:", result["route"])

    print("ANSWER:")

    print(result["final_response"])


# uv run python -m tests.manual_agent_test