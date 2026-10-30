
import uuid
from pathlib import Path

import fitz
import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from transformers import pipeline


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Heritage-RAG",
    page_icon="🏛️",
    layout="wide",
)

APP_TITLE = "Heritage-RAG"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GENERATION_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_DIR = Path("heritage_rag_data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR = DATA_DIR / "chroma_db"

COLLECTION_NAME = "heritage_documents"
MAX_PDF_SIZE = 3 * 1024 * 1024
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150


# -----------------------------
# Load AI models and database
# -----------------------------
@st.cache_resource(show_spinner="Loading AI models and database...")
def load_resources():

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    generator = pipeline(
        task="text-generation",
        model=GENERATION_MODEL,
        tokenizer=GENERATION_MODEL,
        device=-1,
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    return embedding_model, generator, client, collection


# -----------------------------
# PDF extraction and chunking
# -----------------------------
def extract_pdf_pages(pdf_bytes):

    pages = []

    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:

        for page_number, page in enumerate(pdf, start=1):

            text = page.get_text("text").strip()

            if text:
                pages.append({
                    "page": page_number,
                    "text": text,
                })

    return pages


def split_text(
    text,
    chunk_size=CHUNK_SIZE,
    overlap=CHUNK_OVERLAP
):

    text = " ".join(text.split())

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):

        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks


# -----------------------------
# Index a PDF
# -----------------------------
def index_pdf(uploaded_file, embedding_model, collection):

    pdf_bytes = uploaded_file.getvalue()
    filename = Path(uploaded_file.name).name

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise ValueError(
            "This PDF exceeds the 3 MB upload limit."
        )

    pages = extract_pdf_pages(pdf_bytes)

    if not pages:
        raise ValueError(
            "No readable text was found. "
            "This may be a scanned PDF that needs OCR."
        )

    documents = []
    metadatas = []
    ids = []

    document_id = str(uuid.uuid4())

    for page in pages:

        chunks = split_text(page["text"])

        for chunk_number, chunk in enumerate(chunks):

            documents.append(chunk)

            metadatas.append({
                "filename": filename,
                "page": page["page"],
                "document_id": document_id,
                "chunk": chunk_number,
            })

            ids.append(str(uuid.uuid4()))

    if not documents:
        raise ValueError(
            "No text chunks could be created from this PDF."
        )

    embeddings = embedding_model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist(),
    )

    return len(documents), document_id


# -----------------------------
# Retrieve relevant passages
# -----------------------------
def retrieve_context(
    question,
    embedding_model,
    collection,
    top_k=4
):

    if collection.count() == 0:
        return []

    question_embedding = embedding_model.encode(
        [question],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]

    results = collection.query(
        query_embeddings=[question_embedding.tolist()],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    passages = []

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for text, metadata, distance in zip(
        documents, metadatas, distances
    ):

        passages.append({
            "text": text,
            "filename": metadata.get("filename", "Unknown"),
            "page": metadata.get("page", "?"),
            "distance": distance,
        })

    return passages


# -----------------------------
# General knowledge answer
# -----------------------------
def generate_general_answer(question, generator):

    messages = [
        {
            "role": "system",
            "content": (
                "You are Heritage-RAG, a helpful general-purpose "
                "AI assistant. Answer the user's question clearly "
                "and accurately using your general knowledge. "
                "Do not claim that the answer came from an uploaded "
                "document. If you are unsure or the question needs "
                "current information, say so. For conservation or "
                "safety-related decisions, remind the user to verify "
                "important details with reliable sources and "
                "qualified professionals."
            ),
        },
        {
            "role": "user",
            "content": question,
        },
    ]

    output = generator(
        messages,
        max_new_tokens=250,
        do_sample=False,
    )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):

        for message in reversed(generated):

            if message.get("role") == "assistant":
                return message.get("content", "").strip()

        return str(generated[-1].get("content", "")).strip()

    return str(generated).strip()


# -----------------------------
# Generate a PDF-grounded answer
# -----------------------------
def generate_answer(question, passages, generator):

    if not passages:
        return (
            "I could not find relevant information in the "
            "indexed PDFs. Try rephrasing your question, "
            "upload the relevant document, or switch to "
            "General Knowledge mode."
        )

    context_parts = []

    for index, passage in enumerate(passages, start=1):

        context_parts.append(
            f"[Source {index}: {passage['filename']}, "
            f"page {passage['page']}]\n"
            f"{passage['text']}"
        )

    context = "\n\n".join(context_parts)

    messages = [
        {
            "role": "system",
            "content": (
                "You are Heritage-RAG, an assistant for heritage "
                "and monument conservation. Answer using only "
                "the supplied document excerpts. Do not invent "
                "facts, quotations, or citations. If the excerpts "
                "do not contain enough information, clearly say so. "
                "Cite supporting material using the source labels "
                "provided, such as [Source 1]. Offer practical, "
                "careful explanations suitable for conservation "
                "students and heritage professionals."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer based on the excerpts and cite the relevant "
                "source labels. Distinguish documented facts from "
                "any uncertainty."
            ),
        },
    ]

    output = generator(
        messages,
        max_new_tokens=250,
        do_sample=False,
    )

    generated = output[0]["generated_text"]

    if isinstance(generated, list):

        for message in reversed(generated):

            if message.get("role") == "assistant":
                return message.get("content", "").strip()

        return str(generated[-1].get("content", "")).strip()

    return str(generated).strip()


# -----------------------------
# Document management helpers
# -----------------------------
def get_document_names(collection):

    if collection.count() == 0:
        return []

    result = collection.get(include=["metadatas"])

    names = set()

    for metadata in result.get("metadatas", []):

        if metadata:
            names.add(metadata.get("filename", "Unknown"))

    return sorted(names)


def delete_document(filename, collection):

    result = collection.get(
        where={"filename": filename},
        include=["metadatas"],
    )

    ids = result.get("ids", [])

    if ids:
        collection.delete(ids=ids)

    return len(ids)


# -----------------------------
# User interface
# -----------------------------
st.title("🏛️ Heritage-RAG")

st.caption(
    "Ask questions about heritage conservation using "
    "your PDF documents or general AI knowledge."
)


# -----------------------------
# Sidebar
# -----------------------------
with st.sidebar:

    st.header("📚 Document management")

    st.info(
        "Upload text-based PDFs up to 3 MB each. "
        "Scanned PDFs may require OCR before they can be searched."
    )

    uploaded_files = st.file_uploader(
        "Upload heritage documents",
        type=["pdf"],
        accept_multiple_files=True,
    )

    st.divider()

    st.subheader("Indexed documents")

    try:

        embedding_model, generator, client, collection = (
            load_resources()
        )

        document_names = get_document_names(collection)

        if document_names:

            for name in document_names:
                st.write(f"📄 {name}")

            selected_document = st.selectbox(
                "Choose a document to remove",
                options=document_names,
            )

            if st.button(
                "Delete selected document",
                use_container_width=True,
            ):

                deleted_count = delete_document(
                    selected_document,
                    collection,
                )

                st.success(
                    f"Removed {deleted_count} indexed chunks "
                    f"from {selected_document}."
                )

                st.rerun()

        else:

            st.caption("No documents indexed yet.")

        st.caption(
            f"Indexed text chunks: {collection.count()}"
        )

    except Exception as error:

        st.error(
            "Could not load the AI models or database. "
            "Check your installed packages and available memory."
        )

        st.exception(error)
        st.stop()


# -----------------------------
# Process uploaded PDFs
# -----------------------------
if uploaded_files:

    for uploaded_file in uploaded_files:

        upload_key = (
            f"{uploaded_file.name}:"
            f"{uploaded_file.size}"
        )

        if upload_key not in st.session_state:
            st.session_state[upload_key] = "pending"

        if st.session_state[upload_key] == "pending":

            with st.spinner(
                f"Processing {uploaded_file.name}..."
            ):

                try:

                    chunks_added, _ = index_pdf(
                        uploaded_file,
                        embedding_model,
                        collection,
                    )

                    st.session_state[upload_key] = "done"

                    st.success(
                        f"Indexed {uploaded_file.name}: "
                        f"{chunks_added} text chunks."
                    )

                except Exception as error:

                    st.session_state[upload_key] = "failed"

                    st.error(
                        f"Could not process "
                        f"{uploaded_file.name}: {error}"
                    )


# -----------------------------
# Question answering section
# -----------------------------
st.header("💬 Ask a question")

st.caption(
    "Choose whether you want an answer from general AI "
    "knowledge or from your uploaded PDF documents."
)

question_mode = st.radio(
    "Choose answering mode:",
    options=[
        "General knowledge",
        "Uploaded PDFs",
    ],
    horizontal=True,
    help=(
        "General knowledge does not use PDF content. "
        "Uploaded PDFs searches your indexed documents "
        "and displays sources."
    ),
)


# -----------------------------
# Chat history
# -----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for entry in st.session_state.chat_history:

    with st.chat_message(entry["role"]):
        st.markdown(entry["content"])


# -----------------------------
# User question
# -----------------------------
if question_mode == "General knowledge":

    placeholder_text = "Ask anything..."

else:

    placeholder_text = "Ask a question about your uploaded PDFs..."


question = st.chat_input(placeholder_text)


if question:

    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        try:

            if question_mode == "General knowledge":

                with st.spinner(
                    "Preparing a general AI answer..."
                ):

                    answer = generate_general_answer(
                        question,
                        generator,
                    )

                st.caption(
                    "🌐 Answer mode: General AI knowledge"
                )

                st.markdown(answer)

            else:

                with st.spinner(
                    "Searching uploaded PDFs and preparing an answer..."
                ):

                    passages = retrieve_context(
                        question,
                        embedding_model,
                        collection,
                        top_k=4,
                    )

                    answer = generate_answer(
                        question,
                        passages,
                        generator,
                    )

                st.caption(
                    "📚 Answer mode: Uploaded PDF documents"
                )

                st.markdown(answer)

                if passages:

                    st.subheader("📖 Retrieved sources")

                    for index, passage in enumerate(
                        passages,
                        start=1,
                    ):

                        with st.expander(
                            f"Source {index}: "
                            f"{passage['filename']} — "
                            f"page {passage['page']}"
                        ):

                            st.write(passage["text"])

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
            })

        except Exception as error:

            st.error(
                f"Could not answer the question: {error}"
            )


# -----------------------------
# Footer
# -----------------------------
st.divider()

st.caption(
    "Heritage-RAG supports general AI questions and "
    "document-based question answering. Always verify "
    "important conservation decisions against original "
    "documents and qualified professional guidance."
)
