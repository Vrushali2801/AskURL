"""
AskURL - Ask Questions About Web Content
"""
import streamlit as st
import time
from config import PAGE_TITLE, PAGE_ICON
from utils.document_processor import DocumentProcessor
from utils.qa_chain import QAChain

# Page configuration
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout="wide"
)

st.title(f"{PAGE_ICON} {PAGE_TITLE}")
st.markdown("Enter URLs to analyze and ask questions about the content.")

# Initialize session state
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'urls_processed' not in st.session_state:
    st.session_state.urls_processed = False
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize document processor
doc_processor = DocumentProcessor()

# Sidebar for URL input
with st.sidebar:
    st.header("📎 Enter URLs")
    st.markdown("Add URLs to analyze (one per line)")
    
    urls_input = st.text_area(
        "URLs",
        height=200,
        placeholder="https://example.com/article1\nhttps://example.com/article2"
    )
    
    process_button = st.button("🔄 Process URLs", type="primary")
    load_button = st.button("📂 Load Existing Index")
    
    # Main placeholder for progress messages
    main_placeholder = st.empty()
    
    if process_button:
        if urls_input.strip():
            urls = [url.strip() for url in urls_input.split('\n') if url.strip()]
            
            if len(urls) > 0:
                try:
                    # Show progress: Loading
                    main_placeholder.text("Data Loading...Started...✅✅✅")
                    time.sleep(0.5)
                    
                    # Load URLs
                    documents = doc_processor.load_urls(urls)
                    
                    # Show progress: Text Splitting
                    main_placeholder.text("Text Splitter...Started...✅✅✅")
                    time.sleep(0.5)
                    
                    # Split documents
                    chunks = doc_processor.split_documents(documents)
                    
                    # Show progress: Embedding
                    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
                    time.sleep(0.5)
                    
                    # Create vector store
                    vectorstore = doc_processor.create_vectorstore(chunks)
                    
                    # Save the index
                    doc_processor.save_vectorstore(vectorstore)
                    
                    # Store vectorstore and create QA chain
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = QAChain(vectorstore)
                    st.session_state.urls_processed = True
                    st.session_state.chat_history = []
                    
                    main_placeholder.success(f"✅ Processed {len(documents)} documents into {len(chunks)} chunks & saved index!")
                    
                except Exception as e:
                    main_placeholder.error(f"❌ Error processing URLs: {str(e)}")
            else:
                main_placeholder.warning("⚠️ Please enter at least one valid URL")
        else:
            main_placeholder.warning("⚠️ Please enter URLs to process")
    
    if load_button:
        try:
            # Check if index exists
            if doc_processor.index_exists():
                main_placeholder.text("Loading existing index...⏳")
                time.sleep(0.3)
                
                # Load the vectorstore
                vectorstore = doc_processor.load_vectorstore()
                
                if vectorstore:
                    st.session_state.vectorstore = vectorstore
                    st.session_state.qa_chain = QAChain(vectorstore)
                    st.session_state.urls_processed = True
                    st.session_state.chat_history = []
                    
                    main_placeholder.success("✅ Loaded existing index successfully!")
                else:
                    main_placeholder.error("❌ Failed to load index")
            else:
                main_placeholder.warning("⚠️ No saved index found. Please process URLs first.")
        except Exception as e:
            main_placeholder.error(f"❌ Error loading index: {str(e)}")
    
    # Display status
    st.divider()
    if st.session_state.urls_processed:
        st.success("✅ Ready to answer questions")
    else:
        st.info("ℹ️ Add URLs and click 'Process URLs' to start")

# Main area for Q&A
if st.session_state.urls_processed:
    st.header("💬 Ask Questions")
    
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                # Display answer
                st.markdown(message["content"])
                # Display sources if available
                if "sources" in message and message["sources"]:
                    st.divider()
                    st.caption("**Sources:**")
                    sources_list = message["sources"].split("\n")
                    for source in sources_list:
                        if source.strip():
                            st.caption(f"• {source.strip()}")
            else:
                st.markdown(message["content"])
    
    # Input for questions
    question = st.chat_input("Ask a question about the content...")
    
    if question:
        # Display user message
        with st.chat_message("user"):
            st.markdown(question)
        
        # Add to chat history
        st.session_state.chat_history.append({"role": "user", "content": question})
        
        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer with sources using QA chain
                    result = st.session_state.qa_chain.ask_with_sources(question)
                    answer = result.get("answer", "")
                    sources = result.get("sources", "")
                    
                    # Display answer
                    st.markdown(answer)
                    
                    # Display sources
                    if sources:
                        st.divider()
                        st.caption("**Sources:**")
                        sources_list = sources.split("\n")
                        for source in sources_list:
                            if source.strip():
                                st.caption(f"• {source.strip()}")
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating answer: {str(e)}"
                    st.error(error_msg)
                    st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
else:
    st.info("👈 Please add URLs in the sidebar and click 'Process URLs' to begin.")
    
    # Show example
    with st.expander("📖 How to use"):
        st.markdown("""
        ### Getting Started
        
        1. **Add URLs**: Enter one or more URLs in the sidebar (one per line)
        2. **Process**: Click the 'Process URLs' button to load and analyze the content
        3. **Ask Questions**: Once processed, type your questions in the chat input
        4. **Get Answers**: The AI will answer based on the content from the URLs, along with source URLs
        
        ### Features
        
        - ✅ **Persistence**: Processed data is automatically saved for future use
        - 📂 **Load Index**: Click 'Load Existing Index' to reuse previously processed data
        - 🔗 **Source URLs**: Each answer shows which articles it came from
        - 💬 **Chat History**: Keep track of your conversation
        
        ### Example URLs you can try:
        ```
        https://www.moneycontrol.com/news/business/markets/wall-street-rises-as-tesla-soars-on-ai-optimism-11351111.html
        https://www.moneycontrol.com/news/business/tata-motors-launches-punch-icng-price-starts-at-rs-7-1-lakh-11098751.html
        https://www.moneycontrol.com/news/business/tata-motors-mahindra-gain-certificates-for-production-linked-payouts-11281691.html
        ```
        """)

# Footer
st.divider()
st.caption("Built with Streamlit, LangChain, and OpenAI • Inspired by CodeBasics RockyBot")
