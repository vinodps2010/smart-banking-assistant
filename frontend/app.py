import streamlit as st

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="NorthStar Bank - Smart Banking Assistant",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
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
# Global Styling
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       Main Page Background
    ------------------------------------------------------- */

    [data-testid="stAppViewContainer"] {
        
    }

    [data-testid="stMainBlockContainer"] {
        
    }


    /* -------------------------------------------------------
       Sidebar
    ------------------------------------------------------- */

    [data-testid="stSidebar"] {
        
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
       Home Content
    ------------------------------------------------------- */

    .home-card {
        border: 1px solid #D0D0D0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
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
# Home Page
# ============================================================

st.title("🏦 NorthStar Bank")

st.subheader("Smart Banking Assistant")


st.markdown(
    """
    <div class="home-card">

    Welcome to the NorthStar Bank Smart Banking Assistant.

    Your AI-powered banking assistant helps customers with:

    - 🏦 Account related queries
    - 🏠 Loan eligibility and policies
    - 💳 Credit card information
    - 💰 Fixed deposits
    - 📄 Banking product documentation
    - 🗄️ Customer transaction information

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Footer
# ============================================================

st.divider()

st.caption(
    "NorthStar Bank Smart Banking Assistant | "
    "Powered by LangGraph + RAG + PostgreSQL + OpenAI"
)
