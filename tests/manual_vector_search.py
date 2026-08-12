from src.retriever.search import vector_search

query = """
What is the eligibility criteria for home loan?
"""


results = vector_search(query, top_k=5)


print("Results:", len(results))


for r in results:

    print("\n----------------")

    print(r["chunk_type"])

    print(r["source_page"])

    print(r["content"][:300])


# uv run python -m tests.manual_vector_search
