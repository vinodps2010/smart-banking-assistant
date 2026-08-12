from src.database.postgres import test_connection

if __name__ == "__main__":

    result = test_connection()

    print("Database connection successful!")
    print("Database:", result[0])
    print("Version:", result[1])
