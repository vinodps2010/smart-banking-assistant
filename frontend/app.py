import streamlit as st

st.set_page_config(
    page_title="Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
)


# Hide Streamlit default page navigation
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


# Custom Sidebar Navigation
with st.sidebar:

    st.title("🏦 Smart Banking")
    st.caption("AI-Powered Banking Assistant")

    st.divider()

    st.subheader("Features")

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

    st.page_link(
        "pages/analytics.py",
        label="System Analytics",
        icon="📊",
        use_container_width=True,
    )

    st.divider()

    st.caption("Powered by AI • RAG • NL-to-SQL")


# Home Page
st.title("🏦 Smart Banking Assistant")

st.subheader("AI-Powered Multimodal Banking Assistant")

st.write("""
    Ask banking questions using a single intelligent chat interface.

    The LangGraph agent automatically determines whether your question
    requires:

    - 📄 RAG retrieval from banking documents
    - 🗄️ SQL retrieval from the banking database
    - 🔀 Both RAG and SQL
    """)


col1, col2, col3 = st.columns(3)

with col1:
    st.info("📄 **RAG Retrieval**\n\n" "Hybrid search using Vector + FTS.")

with col2:
    st.info("🗄️ **SQL Retrieval**\n\n" "Safe natural language to SQL queries.")

with col3:
    st.info("🔀 **Hybrid Routing**\n\n" "LangGraph decides the best retrieval path.")


st.divider()


st.subheader("🚀 How It Works")

st.markdown("""
    **1. Ask a Question** → Enter your banking query

    **2. LangGraph Classifies** → RAG / RDBMS / BOTH

    **3. Retrieve Information** → PGVector and/or PostgreSQL

    **4. Generate Answer** → Grounded AI response
    """)
