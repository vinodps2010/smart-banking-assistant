from openai import OpenAI

from src.common.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

from src.common.logger import logger

client = OpenAI(api_key=OPENAI_API_KEY)


QUERY_REWRITE_PROMPT = """
You are a query rewriting component for a banking knowledge-base
retrieval system.

Your job is to rewrite a user's query ONLY when the initial retrieval
is weak.

The rewritten query must maximize retrieval of the correct banking
product, policy, or requirement.

IMPORTANT RULES:

1. Preserve the user's original intent.
2. Do NOT answer the question.
3. Do NOT invent banking facts.
4. Do NOT replace an identifiable banking product with generic words
   such as "product" or "application".
5. Preserve known banking product names such as:
   - home loan
   - personal loan
   - fixed deposit
   - credit card
6. Add useful retrieval keywords such as:
   eligibility, criteria, requirements, CIBIL, income,
   interest rate, LTV, tenure, charges, etc.,
   only when supported by the query or retrieved context.
7. Prefer a concise search query rather than a full sentence.
8. If the original query is ambiguous and the product cannot
   be confidently identified from the context, return:

    CLARIFICATION_REQUIRED

    Do not guess a product.
9. Return ONLY the rewritten search query.

Original User Query:
{query}

Initial Retrieved Context:
{context}
"""


def rewrite_query(
    query: str,
    context: str = "",
) -> str:
    """
    Rewrite a weak banking retrieval query.
    """

    logger.info("Query rewrite started")

    prompt = QUERY_REWRITE_PROMPT.format(
        query=query,
        context=context,
    )

    try:

        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        rewritten = (response.choices[0].message.content or "").strip()

        logger.info(
            "Query rewrite completed | result_length=%d",
            len(rewritten),
        )

        if rewritten == "CLARIFICATION_REQUIRED":

            logger.info("Query rewrite requested clarification")

        return rewritten

    except Exception:

        logger.exception("Query rewrite failed")

        raise
