# ERP Chatbot with Image and PDF Analysis

This Streamlit application provides an ERP (Enterprise Resource Planning) chatbot with additional capabilities for image and PDF analysis. It integrates natural language processing, OCR (Optical Character Recognition), and document analysis to offer a versatile tool for various business needs.

## Features

- **Chat Mode**: Interact with an AI assistant for general queries and information.
- **Image Analysis**: Upload and analyze images (up to 5) to extract text and classify bill types.
- **PDF Analysis**: Upload and analyze PDF documents (up to 5) to extract text and generate summaries.
- **Q&A**: Ask questions about the extracted content from images and PDFs.
- **User Authentication**: Secure login system to access the application.

## Requirements

- Python 3.7+
- Streamlit
- OpenAI API key
- EasyOCR
- PyMuPDF
- OpenCV
- Other dependencies listed in `requirements.txt`

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/erp-chatbot.git
   cd erp-chatbot
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Set up the database:
   - Install MySQL and create a database named `streamlit_app_db`.
   - Run the SQL commands provided in `database1.sql` to set up the necessary tables.

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open your web browser and navigate to the provided local URL (usually `http://localhost:8501`).

3. Log in using the predefined credentials or set up new user accounts.

4. Choose a mode (Chat, Image Analysis, or PDF Analysis) from the sidebar.

5. Follow the on-screen instructions to interact with the chatbot, upload images/PDFs, or ask questions about the analyzed content.

## Acknowledgements

- OpenAI for providing the GPT models
- EasyOCR for optical character recognition
- PyMuPDF for PDF processing
- Streamlit for the web application framework
