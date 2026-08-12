import streamlit as st

# from frontend.api_client import api_client
from api_client import api_client

st.set_page_config(
    page_title="System Analytics",
    page_icon="📊",
    layout="wide",
)


st.title("📊 System Analytics")
st.caption("Monitor the Smart Banking Assistant system.")

st.divider()


# Check backend health
if st.button(
    "🔄 Refresh System Status",
    type="primary",
):
    with st.spinner("Checking system status..."):
        health = api_client.health_check()
else:
    health = api_client.health_check()


# System status
status = health.get("status", "offline")


if status.lower() in ["healthy", "online", "ok"]:
    st.success("🟢 System Status: Healthy")
else:
    st.error("🔴 System Status: Offline")


# Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "API Status",
        status.title(),
    )

with col2:
    st.metric(
        "Service",
        health.get(
            "service",
            "N/A",
        ),
    )

with col3:
    st.metric(
        "Mode",
        health.get(
            "mode",
            "API",
        ).upper(),
    )

with col4:
    st.metric(
        "API Version",
        health.get(
            "version",
            "1.0.0",
        ),
    )


st.divider()


# Application components
st.subheader("Application Components")

component_data = [
    {
        "Component": "Streamlit Frontend",
        "Status": "Ready",
    },
    {
        "Component": "FastAPI Backend",
        "Status": ("Mock Mode" if health.get("mode") == "mock" else status.title()),
    },
    {
        "Component": "PostgreSQL Database",
        "Status": "Pending Integration",
    },
    {
        "Component": "RAG Pipeline",
        "Status": "Pending Integration",
    },
    {
        "Component": "NL-to-SQL Engine",
        "Status": "Pending Integration",
    },
]

st.dataframe(
    component_data,
    use_container_width=True,
    hide_index=True,
)


st.divider()


# Health check response
with st.expander("View Health Check Response"):
    st.json(health)


# Error details
if health.get("error"):

    st.divider()

    st.subheader("Error Details")

    st.code(
        health.get("error"),
    )
