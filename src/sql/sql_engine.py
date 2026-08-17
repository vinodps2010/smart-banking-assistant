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

from src.common.config import (
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

from src.common.prompts import SQL_GENERATION_PROMPT
from src.common.schemas import SQLGenerationResponse
from src.database.postgres import get_connection
from src.common.logger import logger

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
# Destructive Intent Detection
# ============================================================


def contains_destructive_intent(
    query: str,
) -> bool:
    """
    Detect user requests that attempt
    destructive database operations.
    """

    destructive_patterns = [
        r"\bdelete\b",
        r"\bremove\b",
        r"\bdrop\b",
        r"\btruncate\b",
        r"\berase\b",
        r"\bclear\b",
        r"\bdestroy\b",
        r"\bmodify\b",
        r"\bupdate\b",
        r"\bchange\b",
    ]

    query_lower = query.lower()

    for pattern in destructive_patterns:

        if re.search(
            pattern,
            query_lower,
        ):

            logger.info("Destructive SQL intent detected")

            return True

    return False


# ============================================================
# SQL Validation
# ============================================================


def validate_sql(
    sql: str,
) -> tuple[bool, str]:
    """
    Validate SQL before execution.

    Rules:
    - SQL cannot be empty.
    - Only SELECT statements are allowed.
    - Multiple SQL statements are rejected.
    - Data modification / DDL operations are rejected.
    """

    if not sql or not sql.strip():

        logger.info("SQL validation failed | reason=empty_sql")

        return (
            False,
            "SQL query cannot be empty.",
        )

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

        logger.info("SQL validation failed | reason=non_select")

        return (
            False,
            "Only SELECT statements are allowed.",
        )

    # --------------------------------------------------------
    # Multiple statements are not allowed
    # --------------------------------------------------------

    if ";" in sql_without_semicolon:

        logger.info("SQL validation failed | reason=multiple_statements")

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

            logger.info(
                "SQL validation failed | forbidden_operation=%s",
                keyword,
            )

            return (
                False,
                f"Forbidden SQL operation detected: {keyword}",
            )

    return (
        True,
        "SQL validation successful.",
    )


# ============================================================
# LIMIT Handling
# ============================================================


def apply_row_limit(
    sql: str,
) -> str:
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

            logger.debug(
                "SQL row limit preserved | limit=%d",
                existing_limit,
            )

            return sql

        logger.info(
            "SQL row limit reduced | original_limit=%d | max_rows=%d",
            existing_limit,
            MAX_ROWS,
        )

        return re.sub(
            r"\bLIMIT\s+\d+",
            f"LIMIT {MAX_ROWS}",
            sql,
            flags=re.IGNORECASE,
        )

    # No LIMIT -> add one.

    logger.debug(
        "SQL row limit added | max_rows=%d",
        MAX_ROWS,
    )

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
    """

    logger.info("SQL execution started :%s", sql)

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    is_valid, validation_message = validate_sql(sql)

    if not is_valid:

        logger.info("SQL execution rejected | validation_failed")

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

        logger.debug("Opening database connection")

        connection = get_connection()

        cursor = connection.cursor()

        # ----------------------------------------------------
        # Execute
        # ----------------------------------------------------

        cursor.execute(
            safe_sql,
            params or (),
        )

        logger.debug("SQL statement executed successfully")

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

        result_rows = [
            dict(
                zip(
                    column_names,
                    row,
                )
            )
            for row in rows
        ]

        # Safety guard.
        result_rows = result_rows[:MAX_ROWS]

        logger.info(
            "SQL execution completed | row_count=%d",
            len(result_rows),
        )

        return {
            "success": True,
            "sql": safe_sql,
            "rows": result_rows,
            "row_count": len(result_rows),
            "error": None,
        }

    except Exception as exc:

        logger.exception("SQL database execution failed")

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

            logger.debug("Database connection closed")


# ============================================================
# SQL Generator
# ============================================================


def generate_sql(
    user_query: str,
) -> dict[str, Any]:
    """
    Convert a natural-language banking question into SQL.
    """

    if not user_query or not user_query.strip():

        logger.info("SQL generation rejected | empty_user_query")

        return {
            "success": False,
            "sql": None,
            "explanation": None,
            "error": "User query cannot be empty.",
        }

    logger.info("SQL generation started")

    try:

        model = ChatOpenAI(
            model=OPENAI_MODEL,
            api_key=OPENAI_API_KEY,
            temperature=0,
        )

        structured_model = model.with_structured_output(SQLGenerationResponse)

        prompt = SQL_GENERATION_PROMPT.format(user_query=user_query)

        response = structured_model.invoke(prompt)

        logger.info("SQL generation completed")

        return {
            "success": True,
            "sql": response.sql,
            "explanation": response.explanation,
            "error": None,
        }

    except Exception:

        logger.exception("SQL generation failed")

        return {
            "success": False,
            "sql": None,
            "explanation": None,
            "error": "SQL generation failed.",
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

    logger.debug("process_sql called")

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

    logger.info("Natural-language SQL pipeline started")

    # --------------------------------------------------------
    # Destructive intent guardrail
    # --------------------------------------------------------

    if contains_destructive_intent(user_query):

        logger.info(
            "Natural-language SQL request blocked " "by destructive-intent guardrail"
        )

        return {
            "success": False,
            "user_query": user_query,
            "sql": None,
            "explanation": None,
            "rows": [],
            "row_count": 0,
            "error": (
                "Destructive operations are not allowed. "
                "The assistant supports read-only banking queries only."
            ),
        }

    # --------------------------------------------------------
    # 1. Generate SQL
    # --------------------------------------------------------

    generation_result = generate_sql(user_query)

    if not generation_result["success"]:

        logger.info(
            "Natural-language SQL pipeline stopped | " "reason=sql_generation_failed"
        )

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

        logger.info("Generated SQL failed validation")

        return {
            "success": False,
            "user_query": user_query,
            "sql": generated_sql,
            "explanation": generation_result["explanation"],
            "rows": [],
            "row_count": 0,
            "error": validation_message,
        }

    logger.info("Generated SQL passed validation")

    # --------------------------------------------------------
    # 3. Execute SQL
    # --------------------------------------------------------

    execution_result = execute_sql(generated_sql)

    logger.info(
        "Natural-language SQL pipeline completed | " "success=%s | row_count=%s",
        execution_result["success"],
        execution_result["row_count"],
    )

    return {
        "success": execution_result["success"],
        "user_query": user_query,
        "sql": execution_result["sql"],
        "explanation": generation_result["explanation"],
        "rows": execution_result["rows"],
        "row_count": execution_result["row_count"],
        "error": execution_result["error"],
    }
