"""
SQL response formatter.

Converts SQL engine output
into user-friendly response.
"""

from src.sql.sql_engine import process_natural_language_query


def format_sql_response(result: dict):

    if result.get("error"):

        return {
            "answer": "I was unable to process your request.",
            "sql": result.get("sql"),
            "rows": [],
            "error": result["error"],
        }

    rows = result.get("rows", [])

    if not rows:

        answer = "No matching records found."

    else:

        answer = f"I found {len(rows)} " "matching records.\n\n"

        for index, row in enumerate(rows[:10], start=1):

            answer += f"{index}. "

            for key, value in row.items():

                answer += f"{key}: {value}, "

            answer += "\n"

    return {"answer": answer, "sql": result.get("sql"), "rows": rows, "error": None}


def answer_sql_query(query: str):

    result = process_natural_language_query(query)

    # print("\n[SQL SERVICE DEBUG]")
    # print(result)

    return format_sql_response(result)
