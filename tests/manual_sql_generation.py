from src.sql.sql_engine import process_natural_language_query


def main():

    query = (
        "Show the transactions for account 1345367 "
        "where the amount is greater than 50000"
    )

    result = process_natural_language_query(query)

    print("\nUser Query:")
    print(result["user_query"])

    print("\nGenerated SQL:")
    print(result["sql"])

    print("\nExplanation:")
    print(result["explanation"])

    print("\nRows:")
    for row in result["rows"]:
        print(row)

    print("\nRow Count:")
    print(result["row_count"])

    print("\nError:")
    print(result["error"])


if __name__ == "__main__":
    main()

#  uv run python tests/manual_sql_generation.py
