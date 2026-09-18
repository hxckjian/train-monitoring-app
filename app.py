import streamlit as st


st.set_page_config(
    page_title="Train Condition Monitoring",
    page_icon="🚆",
    layout="wide",
)

st.title("Train Condition Monitoring")
st.write("Upload sensor data and run a subsystem prediction.")

subsystem = st.selectbox(
    "Select subsystem",
    [
        "Door",
        "ACV",
        "Rail Corrugation",
        "SHM",
    ],
)

uploaded_files = st.file_uploader(
    "Upload input files",
    accept_multiple_files=True,
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} file(s) uploaded for {subsystem}.")
    st.info("The prediction pipeline will be connected here.")
