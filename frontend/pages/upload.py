import streamlit as st

from api_client import api_client

# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="NorthStar Bank - Document Upload",
    page_icon="📄",
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
# Page Styling
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
       Upload Content Card
    ------------------------------------------------------- */

    .upload-card {
        border: 1px solid #D0D0D0;
        border-radius: 12px;
        padding: 20px;
        margin-top: 10px;
        margin-bottom: 20px;
    }


    /* -------------------------------------------------------
       Upload Button Area
    ------------------------------------------------------- */

    [data-testid="stFileUploader"] {
        border: 1px solid #B8CEF0;
        border-radius: 12px;
        padding: 10px;
    }


    /* -------------------------------------------------------
       Metrics
    ------------------------------------------------------- */

    [data-testid="stMetric"] {
        border: 1px solid #D0D0D0;
        border-radius: 10px;
        padding: 10px;
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
# Page Header
# ============================================================

st.title("📄 Document Upload")

st.subheader("NorthStar Bank Knowledge Base")

st.caption(
    "Upload banking knowledge documents for "
    "Docling processing, embedding generation, "
    "and PGVector storage."
)

st.divider()


# ============================================================
# Supported File Types
# ============================================================

SUPPORTED_TYPES = ["pdf"]


# ============================================================
# File Upload
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a banking document",
    type=SUPPORTED_TYPES,
    help="Currently supported format: PDF",
)


# ============================================================
# No File Selected
# ============================================================

if not uploaded_file:

    st.info("📁 Please select a PDF banking document to upload.")

    st.stop()


# ============================================================
# Selected File Information
# ============================================================

st.subheader("Selected File")

file_size_mb = uploaded_file.size / (1024 * 1024)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "File Name",
        uploaded_file.name,
    )


with col2:

    st.metric(
        "File Type",
        uploaded_file.type or "application/pdf",
    )


with col3:

    st.metric(
        "File Size",
        f"{file_size_mb:.2f} MB",
    )


st.divider()


# ============================================================
# Upload Button
# ============================================================

if st.button(
    "🚀 Upload and Process",
    type="primary",
    use_container_width=True,
):

    progress_bar = st.progress(0)

    status_text = st.empty()

    try:

        # ------------------------------------------------------
        # Step 1: Upload
        # ------------------------------------------------------

        status_text.write("📤 Uploading document to FastAPI...")

        progress_bar.progress(20)

        # ------------------------------------------------------
        # Step 2: Backend Ingestion
        # ------------------------------------------------------

        status_text.write(
            "🔍 Processing document with Docling, "
            "chunking and generating embeddings..."
        )

        progress_bar.progress(40)

        result = api_client.upload_document(uploaded_file)

        # ------------------------------------------------------
        # Error Response
        # ------------------------------------------------------

        if not result.get(
            "success",
            False,
        ):

            progress_bar.empty()

            status_text.empty()

            st.error("❌ Upload failed")

            st.error(
                result.get(
                    "error",
                    "Unknown error",
                )
            )

            st.stop()

        # ------------------------------------------------------
        # Get Result Status
        # ------------------------------------------------------

        result_status = result.get(
            "status",
            "completed",
        ).lower()

        # ------------------------------------------------------
        # Duplicate Document
        # ------------------------------------------------------

        if result_status == "already_exists":

            progress_bar.progress(100)

            status_text.write("ℹ️ Document already exists.")

            st.info(
                "ℹ️ This document has already been "
                "ingested. No duplicate chunks were created."
            )

            st.divider()

            st.subheader("Document Information")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Document ID",
                    result.get(
                        "document_id",
                        "N/A",
                    ),
                )

            with col2:

                st.metric(
                    "Chunks Created",
                    result.get(
                        "chunks_created",
                        0,
                    ),
                )

            with col3:

                st.metric(
                    "Status",
                    "Already Exists",
                )

            message = result.get("message")

            if message:

                st.caption(message)

        # ------------------------------------------------------
        # New Document Successfully Ingested
        # ------------------------------------------------------

        else:

            progress_bar.progress(80)

            status_text.write("💾 Storing embeddings in PGVector...")

            progress_bar.progress(100)

            status_text.write("✅ Document processed successfully!")

            st.success("✅ Document uploaded and processed successfully!")

            st.divider()

            st.subheader("Processing Summary")

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Document ID",
                    result.get(
                        "document_id",
                        "N/A",
                    ),
                )

            with col2:

                st.metric(
                    "Chunks Created",
                    result.get(
                        "chunks_created",
                        0,
                    ),
                )

            with col3:

                st.metric(
                    "Status",
                    result.get(
                        "status",
                        "Completed",
                    ).title(),
                )

    # ----------------------------------------------------------
    # Unexpected Client / API Error
    # ----------------------------------------------------------

    except Exception as exc:

        progress_bar.empty()

        status_text.empty()

        st.error("❌ Unexpected error while processing " "the document.")

        st.exception(exc)
