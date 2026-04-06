import streamlit as st
import PyPDF2
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.question_answering import load_qa_chain
from langchain_core.prompts import PromptTemplate

st.set_page_config(page_title="AI Assistant", layout="wide", page_icon="🤖")

st.title("🤖 General Knowledge AI Assistant")
st.markdown("Ask me anything! My memory has been permanently loaded by the developer.")

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
        st.error("Developer Error: The background document database hasn't been uploaded to the server yet.")
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


st.markdown("---")
user_question = st.text_input("Ask a question:")
if user_question:
    if not api_key:
        st.error("❌ Please enter your API Key in the sidebar first.")
    else:
        process_user_question(user_question, api_key, chat_model)
