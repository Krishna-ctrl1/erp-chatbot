# Multifunctional AI Assistant

This project is a comprehensive AI-powered assistant that combines chatbot functionality with image and PDF analysis capabilities. It leverages OpenAI's GPT models, optical character recognition (OCR), and various Python libraries to provide a versatile tool for natural language processing, document analysis, and information extraction.

## Features

- **Chat Interface**: Engage in conversations with an AI-powered assistant.
- **Image Analysis**: Upload and analyze images, extract text, and classify bill types.
- **PDF Analysis**: Extract text from PDF files, including scanned documents.
- **Question Answering**: Ask questions about the content of uploaded images or PDFs.
- **Text Summarization**: Generate concise summaries of extracted text.
- **Multi-file Support**: Process up to 5 images or PDFs simultaneously.
- **User Authentication**: Secure login system to access the main application.

## Components

1. **app.py**: Main Streamlit application combining all features.
2. **chatbot.py**: Standalone chatbot implementation.
3. **imagebot.py**: Image analysis and OCR functionality.
4. **pdfbot.py**: PDF processing and analysis.

## Installation

To set up the Multifunctional AI Assistant, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/your-username/multifunctional-ai-assistant.git
   cd multifunctional-ai-assistant
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root and add the following:
   ```
   OPENAI_API_KEY=your_openai_api_key
   LANGCHAIN_API_KEY=your_langchain_api_key
   ```

## Usage

### Running the Main Application

To launch the main Streamlit application:

```
streamlit run app.py
```

This will start the web interface where you can:
- Log in using predefined credentials
- Chat with the AI assistant
- Upload and analyze images
- Process PDF files
- Ask questions about uploaded content

### Using Individual Components

You can also run the individual components separately:

- Chatbot: `streamlit run chatbot.py`
- Image Analysis: `streamlit run imagebot.py`
- PDF Processing: `streamlit run pdfbot.py`

## Dependencies

The project relies on several key libraries and frameworks:

- Streamlit: For the web interface
- OpenAI and LangChain: For natural language processing
- EasyOCR: For optical character recognition
- PyMuPDF: For PDF processing
- OpenCV and Pillow: For image handling
- NumPy: For numerical operations

Refer to `requirements.txt` for a complete list of dependencies and their versions.

## Contributing

Contributions to the Multifunctional AI Assistant are welcome! Please feel free to submit pull requests, create issues or spread the word.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for providing the GPT models
- The creators and maintainers of all the open-source libraries used in this project

