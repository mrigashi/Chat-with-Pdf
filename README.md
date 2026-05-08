# Chat-with-Pdf
A conversational AI application that lets you upload PDF documents and ask natural language questions about their content. Built with Google Gemini, LangChain, FAISS, and Pinecone.

 Features

Upload multiple PDF files at once
Semantic search using FAISS vector store (local) and Pinecone (cloud)
Conversational Q&A powered by Google Gemini 2.5 Flash
Feedback collection from users (rating + comments)
Fast and accurate responses with context-grounded answers


Tech Stack
LayerTechnologyFrontendStreamlitLLMGoogle Gemini 2.5 Flash (gemini-2.5-flash-lite-preview-06-17)EmbeddingsGoogle Generative AI Embeddings (text-embedding-004)Vector Store (Local)FAISSVector Store (Cloud)PineconePDF ParsingPyPDF2ChunkingLangChain RecursiveCharacterTextSplitterFrameworkLangChain

Getting Started
1. Clone the repository
bashgit clone https://https://github.com/mrigashi/Chat-with-Pdf
cd your-repo-name
2. Install dependencies
bashpip install -r requirements.txt
3. Set up environment variables
Create a .env file in the root directory:
envGOOGLE_API_KEY=your_google_api_key_here
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_INDEX_NAME=your_pinecone_index_name_here

Get your Google API key from Google AI Studio
Get your Pinecone API key from Pinecone Console

4. Run the app
bashstreamlit run app.py

 Project Structure
|-- app.py                  # Main Streamlit application
|-- requirements.txt        # Python dependencies
|-- .env                    # API keys (not committed to GitHub)
|-- .gitignore              # Ignore .env and faiss_index/
|-- faiss_index/            # Auto-generated local vector store
|-- feedback_log.txt        # Auto-generated user feedback log

 How It Works

Upload PDFs via the sidebar
Text Extraction — PyPDF2 extracts raw text from all pages
Chunking — Text is split into overlapping chunks (10,000 chars, 1,000 overlap)
Embedding — Google's text-embedding-004 model generates vector embeddings
Storage — Embeddings saved locally in FAISS + metadata upserted to Pinecone
Q&A — User questions are matched against FAISS index; top chunks fed to Gemini via LangChain
Response — Gemini answers strictly from retrieved context (no hallucination)


 Important Notes

The app answers only from the uploaded PDF context — it will not make up information
FAISS index is saved locally in faiss_index/ — re-upload PDFs if you restart the app
Add faiss_index/ and .env to your .gitignore


Author
Mrigashi 
AI Engineer Intern — Sword Software N Technologies Pvt. Ltd.
