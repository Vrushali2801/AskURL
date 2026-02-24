# 🔗 AskURL - Ask Questions About Web Content

A Streamlit web application that lets you load content from multiple URLs and ask questions using AI-powered retrieval-augmented generation (RAG). Built with LangChain, GPT-4o, and FAISS vector search.

## ✨ Features

- **🔗 Multi-URL Support**: Load and analyze content from multiple web pages simultaneously
- **🧠 RAG-Based Q&A**: Get intelligent answers based on retrieved context from your sources
- **📊 Source Citations**: Every answer shows which URLs it came from for transparency
- **💾 FAISS Persistence**: Automatically saves processed data - no need to reprocess
- **💬 Chat Interface**: Modern chat-style UI with conversation history
- **🏗️ Modular Design**: Clean code structure for easy customization

## 🚀 Quick Start

### Installation

```powershell
# Clone the repository
git clone https://github.com/Vrushali2801/AskURL.git
cd AskURL

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GITHUB_TOKEN=your_github_token_here
```

Get your token from [GitHub Models](https://github.com/marketplace/models) or use your Azure OpenAI credentials.

### Run the Application

```powershell
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 📖 How to Use

### Processing URLs

1. **Enter URLs** in the sidebar text area (one per line)
2. **Click "Process URLs"** to load and analyze the content
   - Progress indicators show: Data Loading → Text Splitting → Embedding
3. **Ask questions** in the chat input
4. **View answers** with source URL citations

### Loading Saved Index

Already processed URLs before? Click **"Load Existing Index"** to instantly reload your previous data without reprocessing.

### Example Workflow

```
URLs to try:
https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-11351111.html
https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-11098751.html

Questions:
- "What is the main topic of these articles?"
- "What companies are mentioned?"
- "Summarize the key points"
```

## 🛠️ Technologies

- **Streamlit** - Web interface with chat components
- **LangChain** - RAG framework and document processing
- **OpenAI GPT-4o** - Language model via Azure AI
- **FAISS** - Vector similarity search and storage
- **Unstructured** - URL content loading and parsing
- **Python-dotenv** - Environment variable management

## ⚙️ Configuration

Edit [config.py](config.py) to customize the application:

```python
# Model Settings
LLM_MODEL = "gpt-4o"                    # Language model
EMBEDDING_MODEL = "text-embedding-3-small"  # Embedding model
LLM_TEMPERATURE = 0                     # Response randomness (0 = deterministic)

# Text Processing
CHUNK_SIZE = 1000                       # Text chunk size for splitting
CHUNK_OVERLAP = 200                     # Overlap between chunks

# Retrieval
RETRIEVAL_K = 3                         # Number of documents to retrieve

# Storage
FAISS_INDEX_PATH = "faiss_store_openai.pkl"  # Vector store location
```

## 🏗️ Project Structure

```
News-Search-Tool/
├── app.py                      # Main Streamlit application
├── config.py                   # Configuration settings
├── requirements.txt            # Python dependencies
├── .env                        # Environment variables (create this)
├── utils/
│   ├── __init__.py
│   ├── document_processor.py   # URL loading, text splitting, FAISS operations
│   └── qa_chain.py             # RAG chain with source attribution
└── faiss_store_openai/         # Saved vector store (auto-generated)
    ├── index.faiss
    └── index.pkl
```

### Module Overview

- **`document_processor.py`**: Handles loading URLs, splitting text into chunks, creating embeddings, and managing FAISS vector store persistence
- **`qa_chain.py`**: Manages the RAG chain for question-answering with source tracking and document retrieval
- **`config.py`**: Centralized configuration for models, API settings, and application parameters

## 💡 How It Works

1. **Load**: URLs are fetched and content is extracted using UnstructuredURLLoader
2. **Split**: Documents are split into manageable chunks (1000 chars with 200 overlap)
3. **Embed**: Text chunks are converted to vector embeddings using OpenAI's embedding model
4. **Store**: Embeddings are stored in a FAISS vector database for fast similarity search
5. **Retrieve**: When you ask a question, the most relevant chunks are retrieved
6. **Generate**: GPT-4o generates an answer based on the retrieved context
7. **Cite**: Source URLs are tracked and displayed with each answer

## 🤝 Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests.
