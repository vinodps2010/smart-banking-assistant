from src.retriever.search import hybrid_search

queries = [
    "home loan eligibility",
    "CIBIL score requirement for home loan",
    "maximum LTV for home loan",
]


for query in queries:

    print("\n" + "=" * 80)
    print(f"QUERY: {query}")
    print("=" * 80)

    results = hybrid_search(
        query=query,
        top_k=5,
        candidate_k=10,
        rrf_k=60,
    )

    print(f"\nFinal results: {len(results)}")

    for rank, result in enumerate(
        results,
        start=1,
    ):

        print("\n" + "-" * 60)

        print(f"Final Rank     : {rank}")
        print(f"Chunk ID        : {result['id']}")
        print(f"Document       : {result['document_name']}")
        print(f"Page           : {result['source_page']}")
        print(f"Chunk Type     : {result['chunk_type']}")
        print(f"Vector Rank    : {result.get('vector_rank')}")
        print(f"FTS Rank       : {result.get('fts_rank')}")
        print(f"RRF Score      : {result.get('rrf_score'):.6f}")

        print("\nContent:")
        print(result["content"][:500])

# uv run python -m tests.manual_hybrid_search
