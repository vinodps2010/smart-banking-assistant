"""
SQL response formatter.

Converts SQL engine output
into user-friendly response.
"""

from src.sql.sql_engine import (
    process_natural_language_query,
)

from src.common.logger import logger


def format_sql_response(
    result: dict,
):
    """
    Convert SQL engine output into a customer-friendly response.

    SQL rows remain structured so the frontend can render
    them in tabular format.
    """

    if result.get("error"):

        logger.info(
            "SQL engine returned an error",
        )

        return {
            "answer": (
                "Unable to process this request. "
                "The Smart Banking Assistant can assist "
                "with banking related queries and has "
                "limited read access."
            ),
            "sql": result.get("sql"),
            "rows": [],
            "error": result["error"],
        }

    rows = result.get(
        "rows",
        [],
    )

    if not rows:

        answer = "No matching records found."

    else:

        # -------------------------------------------------
        # Do not print SQL rows here.
        #
        # Streamlit UI will display rows
        # in tabular format.
        # -------------------------------------------------

        answer = f"I found {len(rows)} " "matching records."

    return {
        "answer": answer,
        "sql": result.get("sql"),
        "rows": rows,
        "error": None,
    }


def answer_sql_query(
    query: str,
):
    """
    Execute a natural-language SQL query and format the result.
    """

    try:

        result = process_natural_language_query(query)

        return format_sql_response(result)

    except Exception:

        logger.exception(
            "SQL query processing failed",
        )

        raise
