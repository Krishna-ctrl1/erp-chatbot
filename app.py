import streamlit as st

# Define login credentials (for demonstration)
credentials = {
    'user1': 'password1',
    'user2': 'password2'
}

# Login function
def login(username, password):
    if credentials.get(username) == password:
        st.session_state['authenticated'] = True
        st.success(f"Welcome, {username}!")
    else:
        st.session_state['authenticated'] = False
        st.error("Invalid credentials. Please try again.")

# Logout function
def logout():
    st.session_state['authenticated'] = False
    st.success("You have been logged out.")

# Login page
def login_page():
    st.title("Login Page")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        login(username, password)

    if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
        st.stop()  # Stop the app execution if not authenticated

# Main app logic
def main_app():
    st.title("Main App")
    st.write("You are logged in and can now access the main app.")

    import openai
    from langchain_openai import ChatOpenAI
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    import os
    from dotenv import load_dotenv
    import cv2
    import easyocr
    import numpy as np
    import fitz  # PyMuPDF
    from PIL import Image



    # Load environment variables
    load_dotenv()

    # Retrieve OpenAI API key securely
    api_key = os.getenv("OPENAI_API_KEY")

    # Langsmith Tracking
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "Combined Chatbot, ImageBot, and PDFBot"

    # Initialize EasyOCR reader
    reader = easyocr.Reader(['en'])

    # Prompt Templates
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to user queries."),
        ("user", "{conversation}\nNew Question: {new_question}")
    ])

    image_qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Please respond to the user queries based on the extracted text from the uploaded image or PDF."),
        ("user", "Question: {question}\nExtracted Text: {extracted_text}")
    ])

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a summarizer. Please summarize the following text."),
        ("user", "Text: {extracted_text}")
    ])

    # Helper functions
    def generate_response(prompt, api_key, engine, **kwargs):
        openai.api_key = api_key
        llm = ChatOpenAI(model=engine, temperature=0.7, max_tokens=150)
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke(kwargs)

    def build_conversation_context(history):
        return "\n".join([f"User: {query}\nAssistant: {response}" for query, response in history])

    def classify_bill_type(extracted_text):
        bill_types = {
            "Restaurant Bill": ["restaurant", "cafe", "dinner", "lunch", "meal", "food"],
            "Gas Bill": ["gas", "fuel", "petrol", "diesel"],
            "Electricity Bill": ["electricity", "power", "electric", "utility"],
            "Water Bill": ["water", "sewer", "wastewater", "utility"],
            "Internet Bill": ["internet", "wifi", "broadband", "cable"],
            "Phone Bill": ["phone", "mobile", "cell", "telecom"],
        }
        
        for bill_type, keywords in bill_types.items():
            if any(keyword in extracted_text.lower() for keyword in keywords):
                return bill_type
        
        return "Unknown Bill Type"

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "extracted_texts" not in st.session_state:
        st.session_state["extracted_texts"] = ""
    if "file_names" not in st.session_state:
        st.session_state["file_names"] = []

    # Streamlit UI
    st.title("🤖 Multifunctional AI Assistant")

    # Sidebar for mode selection
    mode = st.sidebar.selectbox("Choose Mode", ["Chat", "Image Analysis", "PDF Analysis"])

    if mode == "Chat":
        st.subheader("💬 Chat Mode")
        user_input = st.text_input("You:", key="chat_input")
        
        if user_input:
            conversation = build_conversation_context(st.session_state["chat_history"])
            response = generate_response(chat_prompt, api_key, "gpt-4", conversation=conversation, new_question=user_input)
            st.session_state["chat_history"].append((user_input, response))
            st.write(f"Assistant: {response}")

    elif mode == "Image Analysis":
        st.subheader("🖼 Image Analysis Mode")
        uploaded_images = st.file_uploader("Upload Image Files (up to 5)", type=["png", "jpeg", "jpg"], accept_multiple_files=True)
        
        if uploaded_images and len(uploaded_images) <= 5:
            for uploaded_image in uploaded_images:
                st.image(uploaded_image, caption=uploaded_image.name, use_column_width=True)
                image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
                result = reader.readtext(image)
                extracted_text = ' '.join([text[1] for text in result])
                
                if extracted_text:
                    st.success(f"Text extracted from {uploaded_image.name}")
                    bill_type = classify_bill_type(extracted_text)
                    st.write(f"Detected Bill Type: {bill_type}")
                    summary = generate_response(summary_prompt, api_key, "gpt-4", extracted_text=extracted_text)
                    st.write("Summary:", summary)
                    st.session_state["extracted_texts"] += extracted_text + "\n\n"
                    st.session_state["file_names"].append(uploaded_image.name)
                else:
                    st.warning(f"No text could be extracted from {uploaded_image.name}")
        
        elif len(uploaded_images) > 5:
            st.warning("Please upload up to 5 images only.")

    elif mode == "PDF Analysis":
        st.subheader("📄 PDF Analysis Mode")
        uploaded_pdfs = st.file_uploader("Upload PDF Files (up to 5)", type=["pdf"], accept_multiple_files=True)
        
        if uploaded_pdfs and len(uploaded_pdfs) <= 5:
            for uploaded_pdf in uploaded_pdfs:
                st.write(f"Processing: {uploaded_pdf.name}")
                pdf_document = fitz.open(stream=uploaded_pdf.read(), filetype="pdf")
                extracted_text = ""
                
                for page in pdf_document:
                    text = page.get_text()
                    if text.strip():
                        extracted_text += text + " "
                    else:
                        pix = page.get_pixmap()
                        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        img_np = np.array(img)
                        ocr_result = reader.readtext(img_np, detail=0)
                        extracted_text += ' '.join(ocr_result) + " "
                
                if extracted_text.strip():
                    st.success(f"Text extracted from {uploaded_pdf.name}")
                    summary = generate_response(summary_prompt, api_key, "gpt-4", extracted_text=extracted_text)
                    st.write("Summary:", summary)
                    st.session_state["extracted_texts"] += extracted_text + "\n\n"
                    st.session_state["file_names"].append(uploaded_pdf.name)
                else:
                    st.warning(f"No text could be extracted from {uploaded_pdf.name}")
        
        elif len(uploaded_pdfs) > 5:
            st.warning("Please upload up to 5 PDFs only.")

    # Common Q&A section for Image and PDF modes
    if mode in ["Image Analysis", "PDF Analysis"] and st.session_state["extracted_texts"]:
        st.subheader("Ask a question about the extracted content")
        question = st.text_input("Your question:")
        if question:
            answer = generate_response(image_qa_prompt, api_key, "gpt-4", question=question, extracted_text=st.session_state["extracted_texts"])
            st.write("Answer:", answer)

    # Display uploaded file names
    if st.session_state["file_names"]:
        st.sidebar.subheader("Uploaded Files")
        for file_name in st.session_state["file_names"]:
            st.sidebar.write(f"- {file_name}")

    # Display chat history in sidebar
    if st.session_state["chat_history"]:
        st.sidebar.subheader("Chat History")
        for query, response in st.session_state["chat_history"]:
            st.sidebar.text(f"You: {query}")
            st.sidebar.text(f"Assistant: {response}")
            st.sidebar.markdown("---")

# Render the login page if the user is not authenticated
if 'authenticated' not in st.session_state or not st.session_state['authenticated']:
    login_page()
else:
    main_app()

# Add logout option
if st.session_state.get('authenticated'):
    if st.button("Logout"):
        logout()
