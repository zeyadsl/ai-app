import streamlit as st
import PyPDF2
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="AI Document Q&A", layout="wide", page_icon="📄")

st.title("📄 AI Document Q&A Tool (Powered by Groq)")
st.markdown("Upload your PDF document and ask questions about its content! This uses exactly the super-fast Llama-3 model via Groq.")

# Sidebar for API Key and Document Upload
st.sidebar.title("⚙️ Configuration")
if "GROQ_API_KEY" in st.secrets:
    api_key = st.secrets["GROQ_API_KEY"]
    st.sidebar.success("✅ Secure API Key Loaded Automatically!")
else:
    api_key = st.sidebar.text_input("Enter your Groq API Key:", type="password")
    st.sidebar.markdown("[Get your free API Key here](https://console.groq.com/keys)")

st.sidebar.markdown("---")
st.sidebar.title("🤖 Chat Model Config")
chat_model = st.sidebar.selectbox("Select Groq Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768", "gemma2-9b-it"])
st.sidebar.info("💡 Llama-3.1 8B is blazing fast and usually best for general usage!")

st.sidebar.markdown("---")
st.sidebar.title("📁 Document Upload")
pdf_docs = st.sidebar.file_uploader("Upload your PDF or TXT Files", accept_multiple_files=True, type=['pdf', 'txt'])

def get_document_text(docs):
    text = ""
    for doc in docs:
        if doc.name.endswith('.pdf'):
            pdf_reader = PyPDF2.PdfReader(doc)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        elif doc.name.endswith('.txt'):
            text += doc.read().decode('utf-8')
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    # Using completely free local embeddings so we don't rely on external APIs for parsing
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # Save the FAISS index locally so we don't have to recompute it for every question
    vector_store.save_local("faiss_index")

def get_conversational_chain(api_key, chat_model_name):
    prompt_template = """
    You are a highly intelligent and helpful AI assistant. 
    You have been provided with some imported documents (Context) to help you accurately answer the user's question.
    
    If the Context contains relevant information, use it to provide a detailed response.
    However, if the Context does NOT contain the answer, do NOT say "the answer is not in the context". 
    Instead, simply ignore the Context and use your own vast general knowledge to answer the user's question as best as you can!
    
    CRITICAL INSTRUCTION: You MUST provide your final answer in the EXACT SAME LANGUAGE that the user used to ask the Question. If they ask in Arabic, answer in Arabic. If they ask in English, answer in English.

    Context:
    {context}
    
    Question: 
    {question}

    Answer:
    """
    
    model = ChatGroq(model_name=chat_model_name, temperature=0.3, groq_api_key=api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def process_user_question(user_question, api_key, chat_model_name):
    if not os.path.exists("faiss_index"):
        st.error("Please process a document first in the sidebar.")
        return

    # Load local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Load the vector search database
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    chain = get_conversational_chain(api_key, chat_model_name)
    
    with st.spinner("AI is thinking..."):
        try:
            response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
            st.write("### AI Response:")
            st.info(response["output_text"])
        except Exception as e:
            st.error(f"Uh oh! The AI ran into an issue:\n\n{str(e)}")


# Process button logic
if st.sidebar.button("Process Documents"):
    if not pdf_docs:
        st.sidebar.error("❌ Please upload at least one PDF.")
    else:
        with st.sidebar.spinner("Processing documents (this might take a moment)..."):
            raw_text = get_document_text(pdf_docs)
            text_chunks = get_text_chunks(raw_text)
            get_vector_store(text_chunks)
            st.sidebar.success("✅ Documents Processed Successfully!")

st.markdown("---")
user_question = st.text_input("Ask a question about your documents:")
if user_question:
    if not api_key:
        st.error("❌ Please enter your API Key in the sidebar first.")
    else:
        process_user_question(user_question, api_key, chat_model)
