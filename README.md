# ERP Chatbot

## Table of Contents
1. [Introduction](#introduction)
2. [Features](#features)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Code Structure](#code-structure)
7. [Security Considerations](#security-considerations)
8. [Future Improvements](#future-improvements)

## Introduction

This project is an ERP Chatbot built using Streamlit, OpenAI's GPT models, and various other libraries. It combines chat functionality with image and PDF analysis capabilities, providing a versatile tool for text generation, document analysis, and question-answering.

## Features

1. **User Authentication**: Secure login system to protect access to the application.
2. **Chat Mode**: Engage in conversations with an AI assistant powered by OpenAI's GPT models.
3. **Image Analysis**: 
   - Upload and analyze images (up to 5 at a time)
   - Extract text from images using OCR (Optical Character Recognition)
   - Classify bill types based on extracted text
   - Generate summaries of extracted text
4. **PDF Analysis**:
   - Upload and analyze PDF files (up to 5 at a time)
   - Extract text from PDFs, including text embedded in images within the PDF
   - Generate summaries of extracted text
5. **Question-Answering**: Ask questions about the extracted content from images or PDFs
6. **Chat History**: View past conversations in the sidebar
7. **File Upload Tracking**: Display names of uploaded files in the sidebar

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Log in using the predefined credentials:
   - Username: user1, Password: password1
   - Username: user2, Password: password2

4. Choose a mode (Chat, Image Analysis, or PDF Analysis) from the sidebar.

5. Follow the on-screen instructions for each mode to interact with the AI assistant.

## Code Structure

The main application logic is contained in `app.py`. Here's a breakdown of its key components:

1. **Imports and Setup**:
   - The code imports necessary libraries and sets up environment variables.
   - It initializes the EasyOCR reader for text extraction from images.

2. **Authentication System**:
   - `login()` and `logout()` functions handle user authentication.
   - `login_page()` renders the login interface.

3. **Helper Functions**:
   - `generate_response()`: Generates responses using OpenAI's GPT models.
   - `build_conversation_context()`: Constructs the conversation history for the chat mode.
   - `classify_bill_type()`: Determines the type of bill based on extracted text.

4. **Main Application Logic (`main_app()`)**:
   - Implements the core functionality of the three modes: Chat, Image Analysis, and PDF Analysis.
   - Manages file uploads, text extraction, and interaction with the AI models.

5. **Streamlit UI**:
   - Creates the user interface using Streamlit components.
   - Implements sidebar for mode selection and displays chat history and uploaded file names.

6. **Session State Management**:
   - Utilizes Streamlit's session state to persist data across reruns.

## Security Considerations

1. **API Key Protection**: The OpenAI API key is stored in an environment variable to prevent exposure.
2. **User Authentication**: Basic authentication system is implemented, but for production use, a more robust system should be employed.
3. **File Upload Limitations**: The app limits uploads to 5 files to prevent resource abuse.


---

This README provides an overview of the Multifunctional AI Assistant. For more detailed information or support, please contact the project maintainers.
