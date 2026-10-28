
import gradio as gr

def respond(message, history):
    return (
        "Welcome to the Open-Source RAG Chatbot! "
        "In the next step, we'll connect PDF documents "
        "so I can answer questions using their contents.\n\n"
        f"Your question: {message}"
    )

demo = gr.ChatInterface(
    fn=respond,
    title="Open-Source RAG Chatbot",
    description="Ask questions and explore your documents."
)

if __name__ == "__main__":
    demo.launch()
