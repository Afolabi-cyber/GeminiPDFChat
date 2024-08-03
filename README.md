# Interactive PDF Chatbot with Google Gemini

Welcome to the Interactive PDF Chatbot project! This application allows users to upload PDF files and interact with them by asking questions about their content. The application leverages Google’s Gemini Pro model to generate accurate and contextually relevant responses.

## Overview

The Interactive PDF Chatbot project integrates several advanced technologies to offer a seamless and intuitive user experience. It extracts text from uploaded PDF files, indexes the content, and enables users to ask questions, receiving detailed answers based on the document content.

## Features

- **PDF Text Extraction**: Efficiently extracts text from multiple PDF files.
- **Text Chunking**: Splits large blocks of text into manageable chunks for better processing and indexing.
- **FAISS Vector Store**: Utilizes FAISS for creating and managing a local vector index to facilitate quick similarity searches.
- **Generative AI Responses**: Uses Google’s Gemini Pro model to generate detailed and contextually accurate responses to user queries based on the PDF content.

## Technologies Used

- **Streamlit**: A Python library for creating interactive web applications.
- **PyPDF2**: A library for extracting text from PDF documents.
- **FAISS**: A library for efficient similarity search and clustering of dense vectors.
- **Google Generative AI SDK**: Provides state-of-the-art language modeling capabilities.
- **`langchain`**: A library for managing question-answering chains and prompts.
- **Python 3.x**: The programming language used for this project.

## Installation

Follow these steps to set up and run the application:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/Afolabi-cyber/GeminiPDFChat/
   cd https://github.com/Afolabi-cyber/GeminiPDFChat/
