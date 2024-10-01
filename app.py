import os
import numpy as np
import streamlit as st
import pandas as pd
import pytesseract
from sentence_transformers import SentenceTransformer
from PIL import Image
from langchain.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from statsmodels.tsa.arima.model import ARIMA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from dotenv import load_dotenv
import re
import faiss

# Load environment variables
load_dotenv()

# Load Sentence Transformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other models as needed

# Streamlit UI
st.title("ERP Chatbot: Manage Finances, Transactions, and Budgets")

# API URL for Flask backend (optional if using for backend services)
FLASK_API_URL = "http://127.0.0.1:5000"  # Update if your Flask app runs on a different address

# Function to generate embeddings using Sentence Transformers
def generate_embeddings(text):
    return embedding_model.encode(text)

# General Knowledge Chatbot function (You can keep it or remove)
def openai_chatbot(prompt):
    # Placeholder for the general knowledge chatbot functionality
    return "I'm a chatbot but currently unable to process queries."

# File upload widgets
uploaded_pdfs = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
uploaded_images = st.file_uploader("Upload image files (e.g., receipts)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Class to hold document data
class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

documents = []

# Process uploaded PDFs
if uploaded_pdfs:
    for uploaded_file in uploaded_pdfs:
        loader = PyPDFLoader(uploaded_file)
        docs = loader.load()
        for doc in docs:
            documents.append(Document(doc.page_content, {"filename": uploaded_file.name}))

# Process uploaded images (OCR)
if uploaded_images:
    for uploaded_file in uploaded_images:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)
        documents.append(Document(text, {"filename": uploaded_file.name}))

# Text splitting and vector store creation for document QA
if documents:
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    splits = text_splitter.split_documents(documents)

    # Generate embeddings for documents using Sentence Transformers
    doc_embeddings = np.array([generate_embeddings(doc.page_content) for doc in splits])
    
    # Set up FAISS index
    index = faiss.IndexFlatL2(doc_embeddings.shape[1])  # L2 distance
    index.add(doc_embeddings)  # Add the document embeddings to the index

    # Set up conversational retrieval
    qa_chain = RetrievalQA.from_chain_type(llm=openai_chatbot, retriever=index)

    # Initialize session state for conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

# Sidebar for displaying chat history and ERP management
st.sidebar.title("Chat History & ERP Management")
if 'history' in st.session_state:
    for chat in st.session_state.history:
        st.sidebar.write(f"**You:** {chat['user']}")
        st.sidebar.write(f"**Assistant:** {chat['assistant']}")

tabs = ["Chat with Bot", "Expense Tracking", "Invoice Management", "Budgeting", "Advanced Search", "Document Classification"]
selection = st.sidebar.radio("Go to", tabs)

# General Chatbot Functionality
if selection == "Chat with Bot":
    st.subheader("Chat with the Assistant")
    
    user_input = st.text_input("Ask a question (General knowledge or document-related):")
    
    if user_input and st.button("Send"):
        # Check if there are documents loaded for document-based questions
        if documents:
            response = qa_chain.run(user_input)
        else:
            # Fallback to general knowledge chatbot if no documents are uploaded
            response = openai_chatbot(user_input)
        
        # Save conversation in session state
        st.session_state.history.append({"user": user_input, "assistant": response})

        # Display the chat
        st.write(f"**You:** {user_input}")
        st.write(f"**Assistant:** {response}")

# Functions for Extracting Invoice Data
def extract_invoice_number(text):
    match = re.search(r'Invoice Number: (\S+)', text)
    return match.group(1) if match else "INV-001"

def extract_total_amount(text):
    match = re.search(r'Total Amount: \$?(\d+(\.\d{2})?)', text)
    return float(match.group(1)) if match else 0.00

# Document Classification Function
def classify_document(text):
    categories = ['Invoice', 'Contract', 'Report']
    
    training_texts = ["Invoice from Vendor X", "Signed contract", "Financial report for Q3"]
    training_labels = ["Invoice", "Contract", "Report"]

    vectorizer = TfidfVectorizer()
    classifier = SVC()

    X_train = vectorizer.fit_transform(training_texts)
    classifier.fit(X_train, training_labels)

    X_test = vectorizer.transform([text])
    prediction = classifier.predict(X_test)
    return prediction[0]

# Expense Tracking Tab
if selection == "Expense Tracking":
    st.subheader("Track your expenses here.")
    uploaded_receipts = st.file_uploader("Upload Receipts (Images)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    expenses = []

    if uploaded_receipts:
        for receipt in uploaded_receipts:
            image = Image.open(receipt)
            text = pytesseract.image_to_string(image)
            st.write(f"Extracted Text: {text}")
            amount = extract_total_amount(text)
            expenses.append({"Vendor": "Unknown Vendor", "Amount": amount})

        expenses_df = pd.DataFrame(expenses)
        st.write("Expenses Summary")
        st.write(expenses_df)

# Invoice Management Tab
if selection == "Invoice Management":
    st.subheader("Manage invoices here.")
    uploaded_invoices = st.file_uploader("Upload Invoices (PDFs)", type="pdf", accept_multiple_files=True)
    invoices = []

    if uploaded_invoices:
        for invoice in uploaded_invoices:
            loader = PyPDFLoader(invoice)
            docs = loader.load()
            for doc in docs:
                text = doc.page_content  # Extract the text from the document
                invoice_number = extract_invoice_number(text)
                total_amount = extract_total_amount(text)
                invoices.append({"Invoice Number": invoice_number, "Total Amount": total_amount})

        invoices_df = pd.DataFrame(invoices)
        st.write("Invoices Summary")
        st.write(invoices_df)

# Budgeting Tab
if selection == "Budgeting":
    st.subheader("Budget Forecasting with ARIMA")

    transaction_data = {
        'Date': pd.date_range(start='2023-01-01', periods=100),
        'Amount': pd.Series([i + (i % 5) * 10 for i in range(100)])
    }
    transactions_df = pd.DataFrame(transaction_data)

    def forecast_budget(dataframe):
        dataframe['Date'] = pd.to_datetime(dataframe['Date'])
        dataframe.set_index('Date', inplace=True)
        model = ARIMA(dataframe['Amount'], order=(1, 1, 1))
        results = model.fit()
        forecast = results.forecast(steps=12)
        return forecast

    forecasted_values = forecast_budget(transactions_df)
    st.line_chart(forecasted_values)

# Advanced Search Tab
if selection == "Advanced Search":
    st.subheader("Search through your invoices and documents using NLP.")
    query = st.text_input("Enter your search query:")
    if query:
        search_result = qa_chain.run(query)
        st.write(search_result)

# Document Classification Tab
if selection == "Document Classification":
    st.subheader("Classify Documents")
    uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True)

    if uploaded_files:
        for document in uploaded_files:
            loader = PyPDFLoader(document)
            docs = loader.load()  # Load documents properly
            for doc in docs:
                text = doc.page_content  # Extract text for classification
                classification = classify_document(text)
                st.write(f"Classified as: {classification}")
