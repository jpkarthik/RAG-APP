import streamlit as st
from Simple_RAG import SimpleRAG
from Multi_Query_RAG import MultiQueryRAG
from Multi_document_RAG import MultiDocumentRAG
from Structured_Output_RAG import StructuredOutputRAG
from Agentic_RAG import AgenticRAG
from Conversational_RAG import ConversationalRAG
from Hierarchical_RAG import HierarchicalRAG
from chroma_utils import load_pdfs_from_directory, add_documents
import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH")
PDF_Directory = os.getenv("PDF_Directory")
print(PDF_Directory)

# Initialize RAG agents
simple_rag = SimpleRAG()

multi_query_rag = MultiQueryRAG()
multi_document_rag = MultiDocumentRAG()
structured_rag = StructuredOutputRAG(max_history=5)
agentic_rag = AgenticRAG(max_history=5)
conversation_rag = ConversationalRAG(max_history=5)
hierarchical_rag = HierarchicalRAG()

# Streamlit app configuration
st.set_page_config(page_title="Retreival Aggumented Generation Application", layout="wide")
st.title("Retreival Aggumented Generation Application")

# Sidebar for controls
st.sidebar.header("Settings")
rag_type = st.sidebar.selectbox("Select RAG Type", ["Simple", "Multi Query", "Multi Document", "Structured Output", "Agentic", "Conversation", "Hierarchical"])
st.write("Selected RAG Type is "+rag_type)
response_format = st.sidebar.radio("Response Format", ["Raw", "LLM-enhanced"])
show_debug = st.sidebar.checkbox("Show Debug Logs", value=True)  # Default to True for debugging
clear_db = st.sidebar.button("Clear ChromaDB")
upload_pdfs = st.sidebar.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

# Handle PDF uploads
if upload_pdfs:
    directory = PDF_Directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    collections = []
    for uploaded_file in upload_pdfs:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        with open(file_path, "rb") as pdf_file:
            collection, processed = add_documents(pdf_file, uploaded_file.name)
            if collection and processed:
                collections.append(collection)
    st.sidebar.success(f"Processed PDF(s).")
    if collections:
        st.session_state.collections = collections
    else:
        st.sidebar.warning("No valid collections created from uploaded PDFs.")

# Clear ChromaDB if requested
if clear_db:
    import shutil
    shutil.rmtree(CHROMA_DB_PATH, ignore_errors=True)
    if not os.path.exists(CHROMA_DB_PATH):
        os.makedirs(CHROMA_DB_PATH)
    #os.makedirs(CHROMA_DB_PATH)
    st.sidebar.success("ChromaDB cleared.")
    st.session_state.collections = []

# Load or initialize collections with default PDF fallback
if "collections" not in st.session_state:
    directory = PDF_Directory
    _, _, collections = load_pdfs_from_directory(directory)
    st.session_state.collections = collections if collections else []
    if not st.session_state.collections:
        st.warning("No collections loaded. Using default PDF if available. Check logs for details.")

# Main query interface
query = st.text_input("Enter your query", "")
if st.button("Submit Query"):
    if not st.session_state.collections:
        st.error("No document collections available. Ensure PDFs are uploaded or the default PDF is valid. Check debug logs.")
    else:
        with st.spinner("Processing query..."):
            rag_instance = {
                "Simple": simple_rag,
                "Multi Query": multi_query_rag,
                "Multi Document": multi_document_rag,
                "Structured Output": structured_rag,
                "Agentic": agentic_rag,
                "Conversation": conversation_rag,
                "Hierarchical": hierarchical_rag
            }[rag_type]
            fine_tune = response_format == "LLM-enhanced"
            if rag_type in ["Structured Output", "Agentic"]:
                response = rag_instance.structured_output_rag(query, st.session_state.collections, top_k=3, fine_tune=fine_tune) if rag_type == "Structured Output" else rag_instance.agentic_rag(query, st.session_state.collections, top_k=3, fine_tune=fine_tune)
            else:
                response = rag_instance.simple_rag_func(query, st.session_state.collections) if rag_type == "Simple" else \
                           rag_instance.multi_query_rag(query, st.session_state.collections, top_k=3) if rag_type == "Multi Query" else \
                           rag_instance.multi_document_rag(query, st.session_state.collections, top_k=3) if rag_type == "Multi Document" else \
                           rag_instance.conversational_rag(query, st.session_state.collections, top_k=3) if rag_type == "Conversation" else \
                           rag_instance.hierarchical_rag(query, st.session_state.collections) if rag_type == "Hierarchical" else ""
            if not response or "No relevant documents found" in response:
                st.warning("No results found. Ensure PDFs are uploaded or the default PDF is available. Check debug logs.")
            else:
                st.session_state.response = response.split("\n")
                if show_debug:
                    st.session_state.debug = True
                else:
                    st.session_state.debug = False

# Display results
if "response" in st.session_state:
    import re
    import json

    for line in st.session_state.response:
        if line.startswith("Agent Reasoning:"):
            st.subheader(line)
        elif line.startswith("Retrieved Chunks:"):
            with st.expander("Retrieved Chunks"):
                for next_line in st.session_state.response[st.session_state.response.index(line) + 1:]:
                    if next_line.startswith("Structured Output") or next_line.startswith("Error"):
                        break
                    st.text(next_line)
        elif line.startswith("Structured Output") or line.startswith("Error"):
            # Extract JSON from the response
            response_text = "\n".join(st.session_state.response[st.session_state.response.index(line):])
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0).strip()
                try:
                    parsed_json = json.loads(json_str)
                    st.json(parsed_json)
                except json.JSONDecodeError as e:
                    st.error(f"Failed to parse JSON: {str(e)}")
                    st.text(response_text)
            else:
                st.text(response_text)
        elif st.session_state.debug:
            st.text(line)

# Display history (optional)
if st.sidebar.checkbox("Show Conversation History", value=False):
    history_context = multi_query_rag.get_history_context() if rag_type == "Multi Query" else \
                     multi_document_rag.get_history_context() if rag_type == "Multi Document" else \
                     structured_rag.get_history_context() if rag_type == "Structured Output" else \
                     agentic_rag.get_history_context() if rag_type == "Agentic" else \
                     conversation_rag.get_history_context() if rag_type == "Conversation" else \
                     hierarchical_rag.get_history_context() if rag_type == "Hierarchical" else ""
    if history_context:
        st.sidebar.text_area("Conversation History", history_context, height=200)
    else:
        st.sidebar.write("No history available.")

# Add some basic CSS for better layout
st.markdown(
    """
    <style>
    .stExpander {
        background-color: #f0f0f0;
    }
    .stTextInput > div > div > input {
        width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True
)