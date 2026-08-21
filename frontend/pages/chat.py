import streamlit as st
import uuid

from api_client import api_client

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="NorthStar Bank - Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
)


# ============================================================
# Hide Streamlit Default Page Navigation
# ============================================================

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


# ============================================================
# Page Styling
# ============================================================

st.markdown(
    """
    <style>

    /* Main background */

    .stApp {
    }


    /* NorthStar header */

    /* -------------------------------------------------------
       Fixed Streamlit Header Branding
    ------------------------------------------------------- */

    header[data-testid="stHeader"]::before {
        content: "🏦 NorthStar Bank  —  Smart Banking Assistant";
        position: absolute;
        left: 20px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.2px;
        white-space: nowrap;
        pointer-events: none;
        
    }




    /* User message */

    .user-message {

        background-color: #d9fdd3;

        padding: 12px;

        border-radius: 12px;

        margin-bottom: 10px;

    }



    /* Assistant message */

    .assistant-message {

        background-color: #dbeafe;

        padding: 12px;

        border-radius: 12px;

        margin-bottom: 10px;

    }


    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Sidebar Navigation
# ============================================================

with st.sidebar:

    st.title("🏦 NorthStar Bank")

    st.caption("Smart Banking Assistant")

    st.divider()

    st.subheader("Navigation")

    st.page_link(
        "app.py",
        label="Home",
        icon="🏠",
        use_container_width=True,
    )

    st.page_link(
        "pages/chat.py",
        label="Banking Assistant",
        icon="💬",
        use_container_width=True,
    )

    st.page_link(
        "pages/upload.py",
        label="Document Upload",
        icon="📄",
        use_container_width=True,
    )

    st.divider()

    st.caption("Powered by AI • RAG • NL-to-SQL • LangGraph")


# ============================================================
# Session State Initialization
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


# ============================================================
# Helper Functions
# ============================================================


def clear_chat():

    st.session_state.messages = []

    st.session_state.session_id = str(uuid.uuid4())


def display_sources(
    sources,
    route=None,
    confidence_score=None,
    retry_count=None,
):
    if not sources:

        return

    with st.expander(
        "📚 Sources",
        expanded=False,
    ):

        documents = {}

        for source in sources:

            if source.get("source_type") == "database":
                st.markdown(f"🗄️ **{source.get('source_name')}**")
                continue

            document_name = source.get(
                "document_name",
                "Unknown",
            )

            page = source.get(
                "source_page",
                "N/A",
            )

            if document_name not in documents:

                documents[document_name] = set()

            documents[document_name].add(str(page))

        for document, pages in documents.items():

            st.markdown(f"""
                📄 **{document}**

                Pages: {", ".join(sorted(pages))}
                """)

        if route:
            st.caption(f"Route: {route.upper()}")

        if confidence_score is not None:
            st.caption(f"Confidence Score: {confidence_score:.0%}")

        if retry_count is not None:
            st.caption(f"Retry Count: {retry_count}")


def display_sql_result(sql_result):

    if not sql_result:

        return

    st.subheader("🗄️ Query Result")

    st.dataframe(
        sql_result,
        use_container_width=True,
    )


def display_response_details(message):

    confidence = message.get("confidence_score")

    retry_count = message.get(
        "retry_count",
        0,
    )

    if confidence is not None:

        st.caption(f"Confidence Score: {confidence:.0%}")

    # st.caption(f"Retry Count: {retry_count}")

    if message.get("route") in ["rag", "sql", "both"]:
        st.caption(f"Retry Count: {retry_count}")


# ============================================================
# NorthStar Bank Header
# ============================================================

st.title("🏦 NorthStar Bank")

st.subheader("Smart Banking Assistant")

st.caption(
    "Your AI-powered assistant for NorthStar Bank "
    "accounts, loans, credit cards, deposits, "
    "transactions, eligibility, and banking policies."
)


# ============================================================
# Clear Chat Button
# ============================================================

col1, col2 = st.columns([8, 1])


with col2:

    if st.button(
        "🗑️ Clear",
    ):

        clear_chat()

        st.rerun()


# ============================================================
# Display Previous Conversation
# ============================================================

for message in st.session_state.messages:

    if message["role"] == "user":

        with st.chat_message("user"):

            st.markdown(message["content"])

    else:

        with st.chat_message("assistant"):

            st.markdown(message["content"])

            display_sql_result(
                message.get(
                    "sql_result",
                    [],
                )
            )

            display_sources(
                message.get("sources", []),
                message.get("route"),
                message.get("confidence_score"),
                message.get("retry_count"),
            )


# ============================================================
# User Input
# ============================================================

prompt = st.chat_input("Ask your NorthStar Bank question...")


if prompt:

    # --------------------------------------------------------
    # Store User Message
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        status_placeholder = st.empty()

        status_placeholder.info("🤖 Understanding your question...")

        answer_text = ""

        response_metadata = {}

        def stream_response():

            for event in api_client.stream_query(
                query=prompt,
                session_id=st.session_state.session_id,
            ):

                event_type = event.get("type")

                # --------------------------------------------
                # RAG Token Streaming
                # --------------------------------------------

                if event_type == "token":

                    token = event.get(
                        "value",
                        "",
                    )

                    if token:
                        status_placeholder.empty()
                        yield token

                # --------------------------------------------
                # metadata
                # --------------------------------------------

                elif event_type == "metadata":

                    response_metadata.update(
                        {
                            "route": event.get("route"),
                            "sources": event.get("sources", []),
                            "confidence_score": event.get("confidence_score"),
                            "retry_count": event.get("retry_count", 0),
                        }
                    )
                # --------------------------------------------
                # SQL / BOTH / Small Talk
                # --------------------------------------------

                elif event_type == "answer":

                    response_metadata.update(event)

                    yield event.get(
                        "value",
                        "",
                    )

                # --------------------------------------------
                # Error
                # --------------------------------------------

                elif event_type == "error":

                    yield (
                        "\n\n⚠️ "
                        + event.get(
                            "value",
                            "Unknown error",
                        )
                    )

        try:

            # status = st.empty()

            status_placeholder.info("🤖 Processing your request...")

            answer_text = st.write_stream(stream_response())

            status_placeholder.empty()

        except Exception as exc:

            answer_text = "⚠️ Unable to process request.\n\n" + str(exc)

            st.error(answer_text)

    # --------------------------------------------------------
    # Save Assistant Message
    # --------------------------------------------------------

    assistant_message = {
        "role": "assistant",
        "content": answer_text,
        "route": response_metadata.get(
            "route",
            "rag",
        ),
        "query_path": response_metadata.get(
            "route",
            "rag",
        ),
        "source": response_metadata.get(
            "route",
            "rag",
        ),
        "confidence_score": response_metadata.get("confidence_score"),
        "retry_count": response_metadata.get(
            "retry_count",
            0,
        ),
        "sources": response_metadata.get(
            "sources",
            [],
        ),
        "sql_result": response_metadata.get(
            "sql_result",
            [],
        ),
    }

    st.session_state.messages.append(assistant_message)

    st.rerun()
