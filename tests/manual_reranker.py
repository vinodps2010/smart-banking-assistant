from src.retriever.reranker import rerank_documents

documents = [
    {
        "id": 1,
        "content": (
            "NorthStar Bank offers home loans "
            "for purchase, construction and renovation."
        ),
        "source_page": 1,
        "chunk_type": "text",
    },
    {
        "id": 2,
        "content": (
            "Home loan eligibility includes minimum "
            "income, CIBIL score and employment stability."
        ),
        "source_page": 2,
        "chunk_type": "table",
    },
    {
        "id": 3,
        "content": (
            "Fixed deposits earn interest based on " "tenure and applicable rates."
        ),
        "source_page": 4,
        "chunk_type": "text",
    },
]


query = "What is the home loan eligibility criteria?"


results = rerank_documents(
    query=query,
    documents=documents,
    top_k=2,
)


print("\nCohere Reranking Results")
print("========================")

for result in results:

    print("\n-----------------------------")

    print(f"Rank: {result['rerank_rank']}")

    print(f"Score: {result['rerank_score']:.6f}")

    print(f"Page: {result['source_page']}")

    print(f"Type: {result['chunk_type']}")

    print(f"Content: {result['content']}")


# uv run python -m tests.manual_reranker