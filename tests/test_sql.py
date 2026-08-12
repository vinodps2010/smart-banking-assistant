from src.sql.sql_engine import (
    apply_row_limit,
    execute_sql,
    validate_sql,
)

# ============================================================
# Valid SELECT
# ============================================================


def test_valid_select():

    sql = """
        SELECT
            account_id,
            customer_name,
            account_type
        FROM accounts
        ORDER BY account_id
    """

    result = execute_sql(sql)

    assert result["success"] is True
    assert result["row_count"] == 8
    assert len(result["rows"]) == 8


# ============================================================
# INSERT rejected
# ============================================================


def test_insert_rejected():

    sql = """
        INSERT INTO accounts
        (account_id, customer_name)
        VALUES ('9999999', 'Test User')
    """

    result = execute_sql(sql)

    assert result["success"] is False
    assert "SELECT" in result["error"]


# ============================================================
# UPDATE rejected
# ============================================================


def test_update_rejected():

    sql = """
        UPDATE accounts
        SET customer_name = 'Hacker'
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# DELETE rejected
# ============================================================


def test_delete_rejected():

    sql = """
        DELETE FROM accounts
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# DROP rejected
# ============================================================


def test_drop_rejected():

    sql = """
        DROP TABLE accounts
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# ALTER rejected
# ============================================================


def test_alter_rejected():

    sql = """
        ALTER TABLE accounts
        ADD COLUMN test_column TEXT
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# TRUNCATE rejected
# ============================================================


def test_truncate_rejected():

    sql = """
        TRUNCATE TABLE accounts
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# CREATE rejected
# ============================================================


def test_create_rejected():

    sql = """
        CREATE TABLE test_table (
            id INT
        )
    """

    result = execute_sql(sql)

    assert result["success"] is False


# ============================================================
# Multiple statements rejected
# ============================================================


def test_multiple_statements_rejected():

    sql = """
        SELECT * FROM accounts;
        DELETE FROM accounts;
    """

    result = execute_sql(sql)

    assert result["success"] is False
    assert "Multiple SQL statements" in result["error"]


# ============================================================
# Empty SQL rejected
# ============================================================


def test_empty_sql_rejected():

    result = execute_sql("")

    assert result["success"] is False
    assert "empty" in result["error"].lower()


# ============================================================
# LIMIT automatically added
# ============================================================


def test_limit_added():

    sql = """
        SELECT *
        FROM accounts
    """

    safe_sql = apply_row_limit(sql)

    assert "LIMIT 100" in safe_sql.upper()


# ============================================================
# LIMIT > 100 reduced
# ============================================================


def test_limit_reduced():

    sql = """
        SELECT *
        FROM accounts
        LIMIT 500
    """

    safe_sql = apply_row_limit(sql)

    assert "LIMIT 100" in safe_sql.upper()
    assert "LIMIT 500" not in safe_sql.upper()


# ============================================================
# LIMIT <= 100 preserved
# ============================================================


def test_limit_preserved():

    sql = """
        SELECT *
        FROM accounts
        LIMIT 20
    """

    safe_sql = apply_row_limit(sql)

    assert "LIMIT 20" in safe_sql.upper()


# ============================================================
# Parameterized query
# ============================================================


def test_parameterized_query():

    sql = """
        SELECT
            account_id,
            customer_name
        FROM accounts
        WHERE account_id = %s
    """

    result = execute_sql(
        sql,
        params=("1345367",),
    )

    assert result["success"] is True
    assert result["row_count"] == 1
    assert result["rows"][0]["account_id"] == "1345367"


# uv run pytest tests/test_sql.py -v
