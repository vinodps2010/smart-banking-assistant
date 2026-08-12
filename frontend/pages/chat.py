import streamlit as st
import uuid

from api_client import api_client

st.set_page_config(
    page_title="Banking Assistant",
    page_icon="💬",
    layout="wide",
)


# ---------------------------------------------------------
# Hide Streamlit default navigation
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("💬 Smart Banking Assistant")

st.caption(
    "Ask any banking question. The AI agent will automatically "
    "decide whether to use RAG, RDBMS, or both."
)


# ---------------------------------------------------------
# Session State
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())


# ---------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------


def format_route(route):

    if not route:
        return "Unknown"

    route = str(route).lower()

    mapping = {
        "rag": "RAG",
        "sql": "SQL",
        "rdbms": "SQL",
        "both": "RAG + SQL",
    }

    return mapping.get(route, route.upper())


def display_sources(sources):

    if not sources:
        return

    st.markdown("#### Sources")

    for source in sources:

        if isinstance(source, dict):

            document = source.get(
                "document_name",
                "Unknown",
            )

            page = source.get(
                "source_page",
                "N/A",
            )

            chunk_type = source.get(
                "chunk_type",
                "unknown",
            )

            score = source.get("score")

            text = f"• **{document}** " f"— Page {page} " f"— {chunk_type}"

            if score is not None:
                text += f" — score: {score:.3f}"

            st.markdown(text)

        else:
            st.write(f"• {source}")


def display_response_details(message):

    route = message.get("route") or message.get("query_path")

    st.write(f"**Query Path:** {format_route(route)}")

    confidence = message.get("confidence_score")

    if confidence is not None:

        st.write(f"**Confidence Score:** " f"{confidence:.0%}")

    retry_count = message.get(
        "retry_count",
        0,
    )

    st.write(f"**Retry Count:** {retry_count}")

    sql_query = message.get("sql_query")

    if sql_query:

        st.markdown("#### Generated SQL")

        st.code(
            sql_query,
            language="sql",
        )

    sql_result = message.get("sql_result")

    if sql_result:

        st.markdown("#### SQL Result")

        st.dataframe(
            sql_result,
            use_container_width=True,
            hide_index=True,
        )

    sources = message.get("sources") or message.get("citations") or []

    display_sources(sources)


# ---------------------------------------------------------
# Clear Chat
# ---------------------------------------------------------

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()


# ---------------------------------------------------------
# Display Previous Messages
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message["role"] == "assistant":

            with st.expander("🔍 Response Details"):

                display_response_details(message)


# ---------------------------------------------------------
# User Input
# ---------------------------------------------------------

prompt = st.chat_input("Ask your banking question...")


# ---------------------------------------------------------
# Process Query
# ---------------------------------------------------------

if prompt:

    with st.chat_message("user"):

        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your question..."):

            result = api_client.query_assistant(
                query=prompt,
                session_id=(st.session_state.session_id),
            )

        if not result.get(
            "success",
            False,
        ):

            answer = (
                "⚠️ Unable to process your request.\n\n" f"Error: {result.get('error')}"
            )

            st.error(answer)

            assistant_message = {
                "role": "assistant",
                "content": answer,
            }

        else:

            answer = result.get(
                "answer",
                "No answer returned.",
            )

            st.markdown(answer)

            assistant_message = {
                "role": "assistant",
                "content": answer,
                # backend fields
                "route": result.get("route"),
                "query_path": result.get("route"),
                "confidence_score": result.get("confidence_score"),
                "retry_count": result.get(
                    "retry_count",
                    0,
                ),
                "sql_query": result.get("sql_query"),
                "sql_result": result.get("sql_result"),
                "sources": result.get(
                    "sources",
                    [],
                ),
            }

            with st.expander("🔍 Response Details"):

                display_response_details(assistant_message)

        st.session_state.messages.append(assistant_message)
