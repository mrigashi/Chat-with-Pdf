import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from pinecone import Pinecone
import uuid

# Load .env
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Initialize Pinecone SDK v3
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(pinecone_index_name)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

    records = []
    for i, chunk in enumerate(text_chunks):
        records.append({
            "id": str(uuid.uuid4()),
            "values": [0.0],
            "metadata": {
                "text": chunk,
                "chunk_id": i,
                "source": "uploaded_pdf"
            }
        })
    index.upsert(vectors=records)
    st.success("✅ Uploaded PDF chunks to Pinecone successfully.")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not in the provided context, just say "answer is not available in the context", don't make up any information.

    Context: {context}
    Question: {question}
    Answer:
    """

    model = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash-lite-preview-06-17", temperature=0.3
    )
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply:", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("🤖 Streamlit Application")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
            st.success("Done")

    # 4️⃣ Feedback Section
    st.markdown("---")
    st.subheader("💬 Feedback")
    feedback_rating = st.radio(
        "How helpful was the response?",
        ["👍 Helpful", "👎 Not helpful", "😐 Neutral"],
        horizontal=True
    )
    feedback_text = st.text_area("Any suggestions or comments?", placeholder="Write here...")

    if st.button("Submit Feedback"):
        with open("feedback_log.txt", "a", encoding="utf-8") as f:
            f.write(f"Rating: {feedback_rating}\nComments: {feedback_text}\n{'-'*40}\n")
        st.success("✅ Thank you for your feedback!")


if __name__ == "__main__":
    main()
