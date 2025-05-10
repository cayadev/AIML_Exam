import gradio as gr
from Watson_chatbot_backend import ask_watson, get_available_models

def chat_interface(message, model, use_csv, use_pdf):
    try:
        response = ask_watson(
            message=message,
            model_id=model,
            use_csv=use_csv,
            use_pdf=use_pdf
        )
        return response
    except Exception as e:
        return f"Error: {e}"

# List of available models
models = get_available_models()

with gr.Blocks() as demo:
    gr.Markdown("## 🌾 Watson Agricultural Chatbot")

    with gr.Row():
        message_input = gr.Textbox(label="Enter your question", placeholder="Ask about crop conditions, farming, etc.")
    
    with gr.Row():
        model_selector = gr.Dropdown(label="Select Watson Model", choices=models, value=models[0])
    
    with gr.Row():
        use_csv_checkbox = gr.Checkbox(label="Use CSV Crop Descriptions", value=True)
        use_pdf_checkbox = gr.Checkbox(label="Use PDF Documents", value=True)

    with gr.Row():
        submit_button = gr.Button("Ask Watson")
        response_output = gr.Textbox(label="Watson's Response", lines=10)

    submit_button.click(
        fn=chat_interface,
        inputs=[message_input, model_selector, use_csv_checkbox, use_pdf_checkbox],
        outputs=response_output
    )

if __name__ == "__main__":
    demo.launch()
