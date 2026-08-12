"""
SQL Engine for Smart Banking Assistant.

Responsibilities:
1. Validate generated SQL.
2. Enforce read-only SQL execution.
3. Execute safe SELECT queries against PostgreSQL.
4. Limit result sets to 100 rows.
5. Return structured results.

The LLM-based Natural Language -> SQL generation
will be integrated later.
"""

import re
from typing import Any
from langchain_openai import ChatOpenAI

from src.common.config import OPENAI_API_KEY, OPENAI_MODEL
from src.common.prompts import SQL_GENERATION_PROMPT
from src.common.schemas import SQLGenerationResponse

from src.database.postgres import get_connection

# ============================================================
# Configuration
# ============================================================

MAX_ROWS = 100


# SQL operations that must never be executed.
FORBIDDEN_KEYWORDS = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE",
    "MERGE",
}


# ============================================================
# SQL Validation
# ============================================================


def validate_sql(sql: str) -> tuple[bool, str]:
    """
    Validate SQL before execution.

    Rules:
    - SQL cannot be empty.
    - Only SELECT statements are allowed.
    - Multiple SQL statements are rejected.
    - Data modification / DDL operations are rejected.
    """

    if not sql or not sql.strip():
        return False, "SQL query cannot be empty."

    sql = sql.strip()

    # Remove ONE trailing semicolon.
    sql_without_semicolon = sql.rstrip(";").strip()

    # --------------------------------------------------------
    # Only SELECT is allowed
    # --------------------------------------------------------

    if not re.match(
        r"^SELECT\b",
        sql_without_semicolon,
        re.IGNORECASE,
    ):
        return (
            False,
            "Only SELECT statements are allowed.",
        )

    # --------------------------------------------------------
    # Multiple statements are not allowed
    # --------------------------------------------------------

    if ";" in sql_without_semicolon:
        return (
            False,
            "Multiple SQL statements are not allowed.",
        )

    # --------------------------------------------------------
    # Forbidden SQL operations
    # --------------------------------------------------------

    sql_upper = sql_without_semicolon.upper()

    for keyword in FORBIDDEN_KEYWORDS:

        if re.search(
            rf"\b{keyword}\b",
            sql_upper,
        ):
            return (
                False,
                f"Forbidden SQL operation detected: {keyword}",
            )

    return True, "SQL validation successful."


# ============================================================
# LIMIT Handling
# ============================================================


def apply_row_limit(sql: str) -> str:
    """
    Ensure the SQL query does not request more than
    MAX_ROWS rows.

    If the query already has LIMIT:
        - LIMIT <= 100 is preserved.
        - LIMIT > 100 is reduced to 100.

    If no LIMIT exists:
        LIMIT 100 is added.
    """

    sql = sql.strip().rstrip(";").strip()

    # Check whether LIMIT already exists.
    limit_match = re.search(
        r"\bLIMIT\s+(\d+)",
        sql,
        re.IGNORECASE,
    )

    if limit_match:

        existing_limit = int(limit_match.group(1))

        if existing_limit <= MAX_ROWS:
            return sql

        # Replace larger LIMIT with 100.
        return re.sub(
            r"\bLIMIT\s+\d+",
            f"LIMIT {MAX_ROWS}",
            sql,
            flags=re.IGNORECASE,
        )

    # No LIMIT → add one.
    return f"{sql} LIMIT {MAX_ROWS}"


# ============================================================
# SQL Executor
# ============================================================


def execute_sql(
    sql: str,
    params: tuple | None = None,
) -> dict[str, Any]:
    """
    Validate and execute a SQL query.

    Parameters
    ----------
    sql:
        SQL SELECT statement.

    params:
        Optional parameter tuple for parameterized SQL.

    Returns
    -------
    dict
        Structured SQL execution result.
    """

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    is_valid, validation_message = validate_sql(sql)

    if not is_valid:

        return {
            "success": False,
            "sql": sql,
            "rows": [],
            "row_count": 0,
            "error": validation_message,
        }

    # --------------------------------------------------------
    # Apply row limit
    # --------------------------------------------------------

    safe_sql = apply_row_limit(sql)

    connection = None

    try:

        connection = get_connection()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        cursor.execute(
            safe_sql,
            params or (),
        )

        # ----------------------------------------------------
        # Fetch results
        # ----------------------------------------------------

        rows = cursor.fetchall()

        # ----------------------------------------------------
        # Column names
        # ----------------------------------------------------

        column_names = [description[0] for description in cursor.description]

        # ----------------------------------------------------
        # Convert rows to dictionaries
        # ----------------------------------------------------

        result_rows = [dict(zip(column_names, row)) for row in rows]

        # Safety guard.
        result_rows = result_rows[:MAX_ROWS]

        return {
            "success": True,
            "sql": safe_sql,
            "rows": result_rows,
            "row_count": len(result_rows),
            "error": None,
        }

    except Exception as exc:

        return {
            "success": False,
            "sql": safe_sql,
            "rows": [],
            "row_count": 0,
            "error": str(exc),
        }

    finally:

        if connection:
            connection.close()


def generate_sql(user_query: str) -> dict[str, Any]:
    """
    Convert a natural-language banking question into SQL.
    """

    if not user_query or not user_query.strip():

        return {
            "success": False,
            "sql": None,
            "explanation": None,
            "error": "User query cannot be empty.",
        }

    try:

        model = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

        structured_model = model.with_structured_output(SQLGenerationResponse)

        prompt = SQL_GENERATION_PROMPT.format(user_query=user_query)

        response = structured_model.invoke(prompt)

        return {
            "success": True,
            "sql": response.sql,
            "explanation": response.explanation,
            "error": None,
        }

    except Exception as exc:

        return {
            "success": False,
            "sql": None,
            "explanation": None,
            "error": str(exc),
        }


# ============================================================
# Main SQL Engine Interface
# ============================================================


def process_sql(
    sql: str,
    params: tuple | None = None,
) -> dict[str, Any]:
    """
    Main entry point for SQL processing.

    LangGraph's SQL node will eventually call this function.
    """

    return execute_sql(
        sql=sql,
        params=params,
    )


def process_natural_language_query(
    user_query: str,
) -> dict[str, Any]:
    """
    Complete NL -> SQL -> Validation -> Execution pipeline.
    """

    # --------------------------------------------------------
    # 1. Generate SQL
    # --------------------------------------------------------

    generation_result = generate_sql(user_query)

    if not generation_result["success"]:

        return {
            "success": False,
            "user_query": user_query,
            "sql": None,
            "explanation": None,
            "rows": [],
            "row_count": 0,
            "error": generation_result["error"],
        }

    generated_sql = generation_result["sql"]

    # --------------------------------------------------------
    # 2. Validate SQL
    # --------------------------------------------------------

    is_valid, validation_message = validate_sql(generated_sql)

    if not is_valid:

        return {
            "success": False,
            "user_query": user_query,
            "sql": generated_sql,
            "explanation": generation_result["explanation"],
            "rows": [],
            "row_count": 0,
            "error": validation_message,
        }

    # --------------------------------------------------------
    # 3. Execute SQL
    # --------------------------------------------------------

    execution_result = execute_sql(generated_sql)

    return {
        "success": execution_result["success"],
        "user_query": user_query,
        "sql": execution_result["sql"],
        "explanation": generation_result["explanation"],
        "rows": execution_result["rows"],
        "row_count": execution_result["row_count"],
        "error": execution_result["error"],
    }
