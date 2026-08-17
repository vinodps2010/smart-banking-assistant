import uuid

import streamlit as st

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
# Custom Styling
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Main Chat Page Background
    ------------------------------------------------------- */

    [data-testid="stAppViewContainer"] {
    }


    /* -------------------------------------------------------
       Main Content Area
    ------------------------------------------------------- */

    [data-testid="stMainBlockContainer"] {
    }

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


    /* -------------------------------------------------------
       User Message
    ------------------------------------------------------- */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-user"]
    ) {
        background-color: #D6E8FF;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border: 1px solid #A8C9F5;
    }


    /* -------------------------------------------------------
       Assistant Message
    ------------------------------------------------------- */

    [data-testid="stChatMessage"]:has(
        [data-testid="chatAvatarIcon-assistant"]
    ) {
        background-color: #F2F2F2;
        border-radius: 12px;
        padding: 10px 14px;
        margin-bottom: 8px;
        border: 1px solid #BDBDBD;
    }


    /* -------------------------------------------------------
       Sources Expander
    ------------------------------------------------------- */

    [data-testid="stExpander"] {
        border-radius: 10px;
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

    st.subheader("Features")

    # --------------------------------------------------------
    # Home
    #
    # app.py is the Home page itself.
    # --------------------------------------------------------

    st.page_link(
        "app.py",
        label="Home",
        icon="🏠",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Banking Assistant
    # --------------------------------------------------------

    st.page_link(
        "pages/chat.py",
        label="Banking Assistant",
        icon="💬",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Document Upload
    # --------------------------------------------------------

    st.page_link(
        "pages/upload.py",
        label="Document Upload",
        icon="📄",
        use_container_width=True,
    )

    st.divider()

    st.caption("Powered by AI • RAG • NL-to-SQL • LangGraph")


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
# Session State
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "session_id" not in st.session_state:

    st.session_state.session_id = str(uuid.uuid4())


# ============================================================
# Helper Functions
# ============================================================


def is_small_talk(
    message,
):
    route = message.get("route") or message.get("query_path")

    return route and str(route).lower() == "small_talks"


def display_sources(
    sources,
):
    """
    Display unique customer-facing citations.

    Same document names are grouped together.
    Pages are displayed once.
    """

    if not sources:
        return


    documents = {}


    for source in sources:

        if not isinstance(
            source,
            dict,
        ):
            continue


        document = source.get(
            "document_name",
            "Unknown document",
        )


        page = source.get(
            "source_page",
            "N/A",
        )


        if document not in documents:

            documents[document] = set()


        if page:

            documents[document].add(
                str(page)
            )


    if not documents:
        return


    st.markdown(
        "#### Sources"
    )


    for document, pages in documents.items():

        sorted_pages = sorted(
            pages,
            key=lambda x: int(x)
            if x.isdigit()
            else 9999,
        )


        page_text = ", ".join(
            sorted_pages
        )


        st.markdown(
            f"📄 **{document}**  \n"
            f"   Pages: {page_text}"
        )

def display_source_type(
    source,
):
    """
    Display logical source.

    RAG         -> RAG
    SQL         -> SQL
    BOTH        -> BOTH
    SMALL_TALKS -> SMALL_TALKS
    """

    if not source:
        return

    source = str(source).lower()

    mapping = {
        "rag": "RAG",
        "sql": "SQL",
        "both": "BOTH",
        "small_talks": "SMALL_TALKS",
    }

    display_value = mapping.get(
        source,
        source.upper(),
    )

    st.write(f"**Source:** {display_value}")


def display_sql_result(
    sql_result,
):
    """
    Display SQL results in tabular format.
    """

    if not sql_result:
        return

    st.markdown("#### Transaction / Database Results")

    st.dataframe(
        sql_result,
        width="stretch",
        hide_index=True,
    )


def display_response_details(
    message,
):
    """
    Display customer-useful supporting information.

    Shows:
    - Source type
    - Confidence score
    - Retry Count
    - Document citations

    Internal retrieval/reranking details are hidden.
    """

    # --------------------------------------------------------
    # Source type
    # --------------------------------------------------------

    source = message.get("source")

    display_source_type(source)

    # --------------------------------------------------------
    # Confidence Score
    # --------------------------------------------------------

    confidence = message.get("confidence_score")

    if confidence is not None:

        st.write(f"**Confidence Score:** " f"{confidence:.0%}")

    # --------------------------------------------------------
    # Retry Count
    # --------------------------------------------------------

    retry_count = message.get(
        "retry_count",
        0,
    )

    st.write(f"**Retry Count:** {retry_count}")

    # --------------------------------------------------------
    # Document Sources
    # --------------------------------------------------------

    sources = message.get("sources") or message.get("citations") or []

    display_sources(sources)


# ============================================================
# Clear Chat
# ============================================================

if st.button("🗑️ Clear Chat"):

    st.session_state.messages = []

    st.session_state.session_id = str(uuid.uuid4())

    st.rerun()


# ============================================================
# Display Previous Messages
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        # ----------------------------------------------------
        # SQL result table
        # ----------------------------------------------------

        if message["role"] == "assistant":

            sql_result = message.get("sql_result")

            if sql_result:

                display_sql_result(sql_result)

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            if not is_small_talk(message):

                with st.expander("📚 Sources"):

                    display_response_details(message)


# ============================================================
# User Input
# ============================================================

prompt = st.chat_input("Ask your NorthStar Bank banking question...")


# ============================================================
# Process Query
# ============================================================

if prompt:

    # --------------------------------------------------------
    # User Message
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # --------------------------------------------------------
    # Assistant Response
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner("Analyzing your NorthStar Bank question..."):

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

            # ------------------------------------------------
            # Main answer
            # ------------------------------------------------

            st.markdown(answer)

            assistant_message = {
                "role": "assistant",
                "content": answer,
                "route": result.get("route"),
                "query_path": result.get("route"),
                "source": result.get(
                    "source",
                    result.get("route"),
                ),
                "confidence_score": result.get("confidence_score"),
                "retry_count": result.get(
                    "retry_count",
                    0,
                ),
                "sources": result.get(
                    "sources",
                    [],
                ),
                "sql_result": result.get(
                    "sql_result",
                    [],
                ),
            }

            # ------------------------------------------------
            # SQL result table
            # ------------------------------------------------

            sql_result = assistant_message.get("sql_result")

            if sql_result:

                display_sql_result(sql_result)

            # ------------------------------------------------
            # Sources
            # ------------------------------------------------

            if not is_small_talk(assistant_message):

                with st.expander("📚 Sources"):

                    display_response_details(assistant_message)

        # ----------------------------------------------------
        # Store assistant message
        # ----------------------------------------------------

        st.session_state.messages.append(assistant_message)
