"""
Initialize LangGraph PostgreSQL checkpoint tables.

This creates tables required by LangGraph checkpointing.

NOTE:
These are NOT application chat history tables.
No chats/messages tables are created.
"""

from src.agents.checkpointer import (
    create_checkpointer,
)


def main():

    checkpointer, context = create_checkpointer()

    try:

        print("Initializing LangGraph checkpoint tables...")

        checkpointer.setup()

        print("LangGraph checkpoint tables initialized successfully.")

    finally:

        # Close the underlying context manager
        context.__exit__(
            None,
            None,
            None,
        )


if __name__ == "__main__":

    main()
