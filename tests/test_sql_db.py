from src.database.postgres import get_connection


def main():

    connection = get_connection()

    try:

        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                account_id,
                customer_name,
                account_type
            FROM accounts
            ORDER BY account_id;
            """)

        rows = cursor.fetchall()

        print("\nAccounts found:")
        print("-" * 60)

        for row in rows:
            print(row)

        print("-" * 60)
        print(f"Total accounts: {len(rows)}")

        cursor.close()

    finally:
        connection.close()


if __name__ == "__main__":
    main()

# uv run python test_sql_db.py
