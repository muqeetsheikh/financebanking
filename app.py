from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import streamlit as st

# Function to load the model and tokenizer
def load_model_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model_pipeline = pipeline("text-classification", model=model, tokenizer=tokenizer)
    return model_pipeline

# Set up Streamlit app with a finance theme
st.set_page_config(page_title="Finance & Banking AI", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa; color: #333;
    }
    .stTextArea {margin-top: -1.2rem;}
    .stButton {margin-top: 1.2rem;}
    .stFileUploader {margin-top: -1rem;}
    .stSidebar {background-color: #dce3f0;}
    h1 {
        font-size: 32px; font-weight: bold; color: #2c3e50;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize session state for user spaces and history
if 'user_spaces' not in st.session_state:
    st.session_state['user_spaces'] = ["Default"]
if 'selected_space' not in st.session_state:
    st.session_state['selected_space'] = "Default"
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = {"Default": []}

# Sidebar for user spaces
st.sidebar.markdown("### User Spaces")
selected_space = st.sidebar.selectbox(
    "Select or create a space",
    st.session_state['user_spaces']
)
st.session_state['selected_space'] = selected_space

# Function to create a new user space
def create_user_space(space_name):
    if space_name and space_name not in st.session_state['user_spaces']:
        st.session_state['user_spaces'].append(space_name)
        st.session_state['chat_history'][space_name] = []
        st.success(f"Space '{space_name}' created!")
    elif space_name in st.session_state['user_spaces']:
        st.error("Space already exists!")
    else:
        st.error("Space name cannot be empty!")

# User space creation input
space_name = st.sidebar.text_input("New space name")
if st.sidebar.button("Create Space"):
    create_user_space(space_name)

# Model options
model_options = {
    "BERT ": "philschmid/BERT-Banking77",
    "FinBERT": "RashidNLP/Finance-Sentiment-Classification",
    "DistilBERT": "lxyuan/banking-intent-distilbert-classifier"
}
selected_model_name = st.sidebar.radio("Select Model", list(model_options.keys()))

# Load the selected model pipeline
selected_model = model_options[selected_model_name]
model_pipeline = load_model_pipeline(selected_model)

# Display current space in the main area
st.markdown(f"## User Space: {selected_space}")

# Chat history for the selected space
chat_history = st.session_state['chat_history'][selected_space]

# Display chat history
st.markdown("### Analysis History")
for entry in chat_history:
    if entry["role"] == "user":
        st.markdown(f"*User Input:* {entry['content']}")
    elif entry["role"] == "assistant":
        st.markdown(f"*Analysis:* {entry['content']}")
    st.markdown("---")

# User input for text analysis
user_input = st.text_area("Enter your text for analysis", height=150)

# Generate output based on the selected model
if st.button("Analyze"):
    if user_input.strip():
        # Add user input to chat history
        chat_history.append({"role": "user", "content": user_input})
        st.session_state['chat_history'][selected_space] = chat_history

        # Generate assistant response
        results = model_pipeline(user_input)
        if selected_model_name == "Banking Intent (BERT-Banking77)":
            response = (
                f"Predicted Intent: {results[0]['label']} (Confidence: {results[0]['score']:.4f})"
            )
        elif selected_model_name == "Finance Sentiment Analysis":
            response = (
                f"Sentiment: {results[0]['label']} (Confidence: {results[0]['score']:.4f})"
            )
        elif selected_model_name == "Banking Statement Classifier":
            response = (
                f"Classification: {results[0]['label']} (Confidence: {results[0]['score']:.4f})"
            )

        # Add assistant response to chat history
        chat_history.append({"role": "assistant", "content": response})
        st.session_state['chat_history'][selected_space] = chat_history

        # Display response
        st.markdown("### Analysis Result")
        st.markdown(response)
        st.markdown("---")

    else:
        st.error("Please enter some text for analysis.")

# File upload functionality
uploaded_file = st.file_uploader("Upload a file for analysis", type=["txt", "pdf", "docx"])
if uploaded_file:
    st.write("File upload functionality is under development.")
