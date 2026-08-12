import streamlit as st

from api_client import api_client

st.set_page_config(
    page_title="Document Upload",
    page_icon="📄",
    layout="wide",
)


st.title("📄 Document Upload")

st.caption("Upload banking documents for processing and AI-powered retrieval.")

st.divider()


# Supported file types
SUPPORTED_TYPES = [
    "pdf",
    "png",
    "jpg",
    "jpeg",
]


uploaded_file = st.file_uploader(
    "Choose a document",
    type=SUPPORTED_TYPES,
    help="Supported formats: PDF, PNG, JPG and JPEG",
)


if uploaded_file:

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
            uploaded_file.type,
        )

    with col3:
        st.metric(
            "File Size",
            f"{file_size_mb:.2f} MB",
        )

    st.divider()

    # Preview uploaded image
    if uploaded_file.type and uploaded_file.type.startswith("image"):
        st.subheader("Preview")

        st.image(
            uploaded_file,
            caption=uploaded_file.name,
        )

    # Process button
    if st.button(
        "🚀 Upload and Process",
        type="primary",
        use_container_width=True,
    ):

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:

            # Step 1: Upload
            status_text.write("📤 Uploading document...")
            progress_bar.progress(25)

            result = api_client.upload_document(uploaded_file)

            # Step 2: Processing
            status_text.write("🔍 Processing document...")
            progress_bar.progress(70)

            # Check result
            if not result.get("success", False):

                progress_bar.empty()

                st.error("Upload failed: " f"{result.get('error', 'Unknown error')}")

            else:

                # Complete
                status_text.write("✅ Document processed successfully!")

                progress_bar.progress(100)

                st.success("Document uploaded and processed successfully!")

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

        except Exception as e:

            progress_bar.empty()

            st.error(f"Unexpected error: {str(e)}")

else:

    st.info("📁 Please select a banking document to upload.")
