# gradio_app.py# gradio_app.py

import json
import os
import uuid

import gradio as gr
import requests

# --- Configuration ---
BACKEND_URL_DEFAULT = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


# --- Backend Communication ---

def ping_backend(base_url: str) -> dict:
    """Check if the backend is responsive and get its status."""
    try:
        r = requests.get(f"{base_url}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"status": "error", "message": str(e)}


def upload_documents(files, base_url: str):
    """Send document files to the backend for indexing."""
    if not files:
        return "No files selected.", None
    try:
        files_payload = []
        for f in files:
            files_payload.append(("files", (os.path.basename(f.name), open(f.name, "rb"), "application/octet-stream")))

        r = requests.post(f"{base_url}/upload_docs", files=files_payload, timeout=120)
        if r.status_code != 200:
            return f"Upload failed: {r.text}", None

        data = r.json()
        msg = [
            f"**Documents indexed successfully.**",
            f"- Files processed: {data.get('files_processed')}",
            f"- Original documents: {data.get('original_docs')}",
            f"- Chunks added: {data.get('chunks_added')}"
        ]
        return "\n".join(msg), data
    except Exception as e:
        return f"Error: {e}", None


def upload_images(files, base_url: str):
    """Send image files to the backend for OCR and indexing."""
    if not files:
        return "No images selected.", None
    try:
        files_payload = []
        for f in files:
            files_payload.append(("files", (os.path.basename(f.name), open(f.name, "rb"), "application/octet-stream")))

        r = requests.post(f"{base_url}/upload_images", files=files_payload, timeout=120)
        if r.status_code != 200:
            return f"Upload failed: {r.text}", None

        data = r.json()
        msg = [
            f"**Images OCR'd and indexed successfully.**",
            f"- Files processed: {data.get('files_processed')}",
            f"- Original documents: {data.get('original_docs')}",
            f"- Chunks added: {data.get('chunks_added')}"
        ]
        return "\n".join(msg), data
    except Exception as e:
        return f"Error: {e}", None


# --- UI Callbacks & Helpers ---

def format_citations(citations):
    """Format citation data into markdown."""
    if not citations:
        return "No citations found."
    lines = []
    for c in citations:
        doc_id = c.get("doc_id", "?")
        page = c.get("page", "?")
        source = c.get("source", "?")
        kind = c.get("kind", "")
        suffix = f" ({kind})" if kind else ""
        lines.append(f"- `[{doc_id} p.{page}]` {source}{suffix}")
    return "\n".join(lines)


def chat_call(message, history, session_id, top_k, base_url: str):
    """Handle the user's chat message and return the response."""
    if not message or not message.strip():
        # No change needed to history if the message is empty
        return history, session_id, "No query entered.", ""

    # Append the user's message to the history in the correct format
    history.append({"role": "user", "content": message})

    payload = {"question": message, "top_k": int(top_k)}
    if session_id:
        payload["session_id"] = session_id

    try:
        r = requests.post(f"{base_url}/chat", json=payload, timeout=60)
        if r.status_code != 200:
            err = r.text
            # Append the error message as the assistant's response
            history.append({"role": "assistant", "content": f"Error: {err}"})
            return history, session_id, "—", ""

        data = r.json()
        sid = data.get("session_id") or session_id
        answer = data.get("answer", "")
        citations = data.get("citations", [])
        standalone_q = data.get("standalone_question", message)

        # Append the assistant's successful response
        history.append({"role": "assistant", "content": answer or "(No answer received)"})
        citations_md = format_citations(citations)

        return history, sid, citations_md, standalone_q

    except Exception as e:
        # Append the exception as the assistant's response
        history.append({"role": "assistant", "content": f"Error: {e}"})
        return history, session_id, "—", ""


def clear_session():
    """Clear chat history and create a new session ID."""
    return [], str(uuid.uuid4())


# --- Gradio UI Definition ---

with gr.Blocks(title="RAG Chat Assistant", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# RAG Assistant\n*Upload PDFs/images and chat with your documents. Answers include sources.*")

    with gr.Row():
        backend_url = gr.Textbox(value=BACKEND_URL_DEFAULT, label="Backend URL", interactive=True)
        status_box = gr.Markdown("")

    def refresh_status(url):
        info = ping_backend(url)
        if info.get("status") == "ok":
            emb = info.get("embedding_backend", "?")
            return f"✅ **Backend Connection OK** — Embeddings: `{emb}`"
        return f"⚠️ **Backend Connection Failed:** `{json.dumps(info)}`"

    refresh_btn = gr.Button("Check Backend Status")
    refresh_btn.click(refresh_status, inputs=backend_url, outputs=status_box)

    with gr.Tab("1. Upload Files"):
        gr.Markdown("### Upload PDFs, Text Files, and Images\nDrag & drop files, then click the corresponding button to index them.")
        with gr.Row():
            docs_uploader = gr.File(label="PDF/TXT/Markdown", file_count="multiple", file_types=[".pdf", ".txt", ".md"])
            images_uploader = gr.File(label="Images (PNG/JPG/TIFF)", file_count="multiple", file_types=["image"])
        with gr.Row():
            upload_docs_btn = gr.Button("Index Documents", variant="primary")
            upload_imgs_btn = gr.Button("Index Images", variant="primary")
        
        upload_output = gr.Markdown()

        upload_docs_btn.click(
            fn=upload_documents,
            inputs=[docs_uploader, backend_url],
            outputs=[upload_output, gr.State()],
        )
        upload_imgs_btn.click(
            fn=upload_images,
            inputs=[images_uploader, backend_url],
            outputs=[upload_output, gr.State()],
        )

    with gr.Tab("2. Chat"):
        session_state = gr.State(value=str(uuid.uuid4()))
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(height=500, type="messages", label="Chat")
                # Add elem_id so we can bind a key listener
                query = gr.Textbox(
                    placeholder="Ask a question about your documents...",
                    label="Message",
                    lines=3,
                    elem_id="chat_input"
                )
                send_btn = gr.Button("Send", variant="primary", elem_id="send_btn")
            with gr.Column(scale=1):
                citations_panel = gr.Markdown(label="Citations")
                with gr.Accordion("Advanced Settings", open=False):
                    top_k = gr.Slider(1, 10, value=5, step=1, label="Top-K Documents")
                    standalone_q_box = gr.Textbox(label="Standalone Question (Condensed by LLM)", interactive=False)
                new_session_btn = gr.Button("Clear Chat & Start New Session")

        def on_send(user_msg, history, session_id, k, url):
            history, new_sid, citations_md, standalone_q = chat_call(user_msg, history, session_id, k, url)
            return history, new_sid, citations_md, standalone_q

        # Enter submits, then clear the box
        query.submit(
            on_send,
            inputs=[query, chatbot, session_state, top_k, backend_url],
            outputs=[chatbot, session_state, citations_panel, standalone_q_box],
        ).then(lambda: "", inputs=None, outputs=query)

        # Send button does the same
        send_btn.click(
            on_send,
            inputs=[query, chatbot, session_state, top_k, backend_url],
            outputs=[chatbot, session_state, citations_panel, standalone_q_box],
        ).then(lambda: "", inputs=None, outputs=query)

        new_session_btn.click(
            fn=clear_session,
            inputs=None,
            outputs=[chatbot, session_state]
        ).then(
            lambda: ("Citations will appear here.", "...", ""),
            inputs=None,
            outputs=[citations_panel, standalone_q_box, query]
        )

    # Initial status check on load
    demo.load(refresh_status, inputs=backend_url, outputs=status_box)
    
    # Bind Enter (no Shift) to Send button. Shift+Enter keeps newline.
    demo.load(
        js="""
        () => {
            const box = document.querySelector('#chat_input textarea');
            const btn = document.querySelector('#send_btn');
            if (!box || !btn) return;
            // Avoid attaching multiple times
            if (box.dataset.bound === "1") return;
            box.dataset.bound = "1";
            
            box.addEventListener('keydown', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    btn.click();
                }
            });
        }
        """
    )


if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("FE_PORT", "7860")),
        share=False
    )