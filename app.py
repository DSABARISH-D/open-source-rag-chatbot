import os
import pymupdf
import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

CHUNK_WORDS = 120   # words per chunk
OVERLAP = 30        # words shared between chunks
TOP_K = 3           # number of passages to show
MIN_SCORE = 0.05    # below this, say "not found"


def make_chunks(path):
    """Read a PDF and split it into small overlapping text chunks."""
    name = os.path.basename(path)
    chunks = []
    with pymupdf.open(path) as doc:
        for page_no, page in enumerate(doc, start=1):
            words = page.get_text().split()
            step = CHUNK_WORDS - OVERLAP
            for i in range(0, len(words), step):
                piece = " ".join(words[i:i + CHUNK_WORDS])
                if len(piece.split()) >= 15:
                    chunks.append({"text": piece, "source": f"{name}, page {page_no}"})
    return chunks


def load_pdfs(files):
    """Called when the user uploads PDFs: build the search index."""
    if not files:
        return None, "Please choose at least one PDF."
    chunks = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        chunks.extend(make_chunks(path))
    if not chunks:
        return None, "No readable text found. Scanned PDFs (images) are not supported."
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([c["text"] for c in chunks])
    store = {"chunks": chunks, "vectorizer": vectorizer, "matrix": matrix}
    return store, f"Ready: {len(files)} PDF(s), {len(chunks)} passages indexed. Ask your question below."


def respond(message, history, store):
    if not store:
        return "Please upload a PDF first, then click 'Process PDFs'."
    query = store["vectorizer"].transform([message])
    scores = cosine_similarity(query, store["matrix"]).ravel()
    best = scores.argsort()[::-1][:TOP_K]
    best = [i for i in best if scores[i] >= MIN_SCORE]
    if not best:
        return "I could not find anything about that in your PDFs. Try different words."
    parts = []
    for rank, i in enumerate(best, start=1):
        c = store["chunks"][i]
        parts.append(f"**{rank}. {c['source']}**\n{c['text']}")
    return "Most relevant passages from your PDFs:\n\n" + "\n\n".join(parts)


with gr.Blocks(title="Open-Source RAG Chatbot") as demo:
    gr.Markdown("# Open-Source RAG Chatbot\nUpload PDFs, then ask questions about them.")
    store = gr.State(None)
    files = gr.File(label="Upload PDF files", file_count="multiple", file_types=[".pdf"])
    status = gr.Markdown("")
    gr.Button("Process PDFs", variant="primary").click(load_pdfs, files, [store, status])
    gr.ChatInterface(fn=respond, additional_inputs=[store])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
    )
