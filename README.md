# Streamlit PDF Query Bot

This project is a Streamlit web application that enables users to upload PDF files and interact with their content through a chat interface. The application processes the PDF content using a language model (LLM) and a vector database to provide relevant answers to user queries.

## Features

- **PDF Upload:** Users can upload PDF files, which are processed by the application.
- **Text Extraction:** Extracts text from each page of the PDF.
- **Text Splitting:** The extracted text is split into manageable chunks using `RecursiveCharacterTextSplitter`.
- **Vector Store:** The text chunks are embedded into vectors using Hugging Face embeddings and stored in a Chroma vector database.
- **Question Answering:** The app uses the Groq language model to provide answers to user queries based on the content of the PDF.
- **Conversation Memory:** Maintains a memory of the conversation for context-aware interactions.

## Technologies Used

- **Streamlit:** Framework for building the web application interface.
- **LangChain:** Provides text splitting, embedding, and retrieval capabilities.
- **Hugging Face:** Used for generating text embeddings.
- **Chroma:** Vector store for persisting embedded text data.
- **Groq LLM:** Language model used for generating answers.
- **PyMuPDF (Fitz):** For reading and processing PDF files.
- **dotenv:** For managing environment variables.

## Installation

To run this project locally, follow these steps:

1. **Clone the repository:**

    ```bash
    git clone https://github.com/DhushanthanK/Streamlit-PDFQueryBot.git
    cd Streamlit-PDFQueryBot
    ```

2. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4. **Set up environment variables:**

    Create a `.env` file in the project root and add your Groq API key:

    ```env
    groq_API_KEY=your_groq_api_key_here
    ```

5. **Run the Streamlit app:**

    ```bash
    streamlit run PDFQueryBot.py
    ```

    This will start the Streamlit server and open the app in your default web browser.

## Usage

1. **Upload a PDF:**
   - Use the file uploader widget to upload a PDF file.

2. **Ask Questions:**
   - Type your question in the chat input box. The bot will provide relevant answers based on the content of the uploaded PDF.

3. **Review Conversation:**
   - The chat history is retained, allowing you to ask follow-up questions that consider previous interactions.

## Project Structure

- `app.py`: Main Streamlit application file containing the logic for PDF processing, vector store creation, and interaction with the Groq LLM.
- `requirements.txt`: Lists all the Python dependencies required to run the application.
- `.env`: (Optional) Environment file to manage API keys and other sensitive configurations securely.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contributing

If you wish to contribute to this project, feel free to submit a pull request or open an issue to discuss potential improvements.