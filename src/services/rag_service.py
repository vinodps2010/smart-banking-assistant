"""
RAG service.

Retrieves relevant banking knowledge-base chunks
and generates a grounded answer using OpenAI.
"""

from openai import OpenAI
from dotenv import load_dotenv

from src.retriever.search import vector_search
from src.common.config import OPENAI_API_KEY, OPENAI_MODEL

load_dotenv()

client = OpenAI(api_key=OPENAI_API_KEY)


def _build_context(chunks):
    """
    Build a structured context block for the LLM.

    Including metadata in the context helps the model
    understand where each piece of information came from.
    """

    context_parts = []

    for index, chunk in enumerate(chunks, start=1):

        context_parts.append(f"""
SOURCE {index}
Document: {chunk.get("document_name")}
Page: {chunk.get("source_page")}
Chunk Type: {chunk.get("chunk_type")}
Similarity Score: {chunk.get("score")}

Content:
{chunk.get("content", "")}
""".strip())

    return "\n\n---\n\n".join(context_parts)


def answer_rag_query(query: str):
    """
    Retrieve relevant knowledge-base chunks and
    generate a grounded banking answer.
    """

    chunks = vector_search(
        query,
        top_k=8,
    )

    if not chunks:

        return {
            "answer": (
                "I could not find relevant information "
                "in the banking knowledge base."
            ),
            "sources": [],
        }

    context = _build_context(chunks)

    prompt = f"""
You are a banking assistant for Northstar Bank.

Your task is to answer banking questions using the supplied
knowledge-base content.

The retrieved content may contain:
- document headings
- section titles
- tables
- related paragraphs

Use these together to understand the meaning of the information.
IMPORTANT RULES:

1. Answer only using the provided banking knowledge-base context.

2. Use document structure and section headings to understand context.
   For example:
   - "SECTION 1: HOME LOAN PRODUCTS"
   - "1.3 Eligibility Criteria"
   together indicate that the eligibility information belongs to
   home loan products.

3. Combine information from related chunks, including:
   - section headings
   - text paragraphs
   - tables

4. If a section heading identifies the product area and the following
   content contains eligibility parameters, treat them as belonging
   to that product.

5. Present the answer clearly using bullets or tables.

6. Do not mention retrieval, embeddings, similarity scores,
   or context limitations unless information is truly missing.

7. Do not refuse an answer merely because the exact product name
   is not repeated in every chunk.
   
KNOWLEDGE-BASE CONTEXT:

{context}

USER QUESTION:

{query}
"""

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": chunks,
    }
