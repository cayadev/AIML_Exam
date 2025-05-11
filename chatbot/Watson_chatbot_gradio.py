import gradio as gr
from Watson_chatbot_backend import ask_watson, get_available_models, generate_pdf_content
from Watson_chatbot_judge import judge_response, format_judge_result
import tempfile
import os
import re
import time
from bs4 import BeautifulSoup
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListItem, ListFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT

def chat_interface(message, model, use_csv, use_pdf, use_judge):
    try:
        # Start timing the response
        start_time = time.time()
        
        # Get response from Watson
        response = ask_watson(
            message=message,
            model_id=model,
            use_csv=use_csv,
            use_pdf=use_pdf
        )
        
        # Calculate response time
        response_time = time.time() - start_time
        
        # Check if this is a formatted list response
        if "<FORMATTED_LIST>" in response and "</FORMATTED_LIST>" in response:
            # Remove the formatting tags for display
            clean_response = response.replace("<FORMATTED_LIST>", "").replace("</FORMATTED_LIST>", "")
            is_list = True
        else:
            clean_response = response
            is_list = False
        
        # Get judge evaluation if requested
        judge_evaluation = ""
        if use_judge:
            try:
                # Call the judge function
                judge_result = judge_response(message, clean_response)
                if judge_result["success"]:
                    judge_evaluation = format_judge_result(judge_result)
                else:
                    judge_evaluation = "⚠️ Judge evaluation failed. Please check the model configuration."
            except Exception as e:
                judge_evaluation = f"⚠️ Judge evaluation error: {str(e)}"
        
        # Add response time to judge evaluation or create a new message if no judge
        latency_info = f"\n\n⏱️ **Response Time**: {response_time:.2f} seconds"
        if judge_evaluation:
            judge_evaluation += latency_info
        else:
            judge_evaluation = latency_info.strip()
        
        return clean_response, is_list, judge_evaluation
    except Exception as e:
        return f"Error: {e}", False, ""

def create_pdf(response):
    """Create a PDF file from the response and return the path to the file."""
    try:
        # Generate HTML content
        html_content = generate_pdf_content(response)
        
        # Create a temporary file
        temp_dir = tempfile.gettempdir()
        pdf_path = os.path.join(temp_dir, "watson_response.pdf")
        
        # Create PDF using ReportLab (which handles Unicode properly)
        doc = SimpleDocTemplate(pdf_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create custom styles
        title_style = ParagraphStyle(
            'TitleStyle',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12
        )
        
        normal_style = styles['Normal']
        
        # Create a list style
        list_item_style = ParagraphStyle(
            'ListItemStyle',
            parent=normal_style,
            leftIndent=20,
            firstLineIndent=0
        )
        
        list_bullet_style = ParagraphStyle(
            'ListBulletStyle',
            parent=normal_style,
            leftIndent=30,
            firstLineIndent=0
        )
        
        # Parse HTML content
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Build the PDF content
        elements = []
        
        for element in soup.find_all(['h1', 'p', 'ol', 'li']):
            if element.name == 'h1':
                elements.append(Paragraph(element.text, title_style))
                elements.append(Spacer(1, 12))
            elif element.name == 'p':
                elements.append(Paragraph(element.text, normal_style))
                elements.append(Spacer(1, 6))
            elif element.name == 'li':
                # Check if there's a strong tag for the title
                strong_tag = element.find('strong')
                if strong_tag:
                    title = strong_tag.text
                    description = element.text.replace(title, '', 1).strip()
                    elements.append(Paragraph(f"<b>{title}</b>", list_item_style))
                    elements.append(Paragraph(description, list_bullet_style))
                else:
                    elements.append(Paragraph(element.text, list_item_style))
                elements.append(Spacer(1, 3))
        
        # Build the PDF
        doc.build(elements)
        return pdf_path
    except Exception as e:
        print(f"Error creating PDF: {e}")
        return None

# List of available models
models = get_available_models()

with gr.Blocks() as demo:
    gr.Markdown("## 🌾 FarmWise - Your Agricultural Chatbot")

    with gr.Row():
        # Left panel - Model selection and options
        with gr.Column(scale=1):
            gr.Markdown("### Model & Settings")
            model_selector = gr.Dropdown(
                label="Select Model", 
                choices=models, 
                value=models[0]
            )
            
            use_csv_checkbox = gr.Checkbox(label="Use CSV Crop Descriptions", value=True)
            use_pdf_checkbox = gr.Checkbox(label="Use PDF Documents", value=True)
            use_judge_checkbox = gr.Checkbox(label="Enable Judge Evaluation", value=True)
            
            # Judge evaluation output under the checkboxes
            judge_output = gr.Markdown(label="Evaluation", visible=True)
            
            # PDF download button (initially hidden)
            with gr.Row(visible=False) as pdf_row:
                download_button = gr.Button("Download as PDF")
                pdf_output = gr.File(label="Download PDF")
        
        # Right panel - Question and response
        with gr.Column(scale=2):
            gr.Markdown("### Ask Your Question")
            message_input = gr.Textbox(
                label="", 
                placeholder="Ask about crop conditions, farming, etc.",
                lines=3
            )
            submit_button = gr.Button("Ask FarmWise", variant="primary")
            
            gr.Markdown("### FarmWise's Response")
            response_output = gr.Textbox(label="", lines=20)
    
    # Hidden state
    is_list_state = gr.State(False)
    
    # Function to update UI based on response type
    def update_ui(response, is_list, judge_evaluation):
        return (
            response, 
            is_list, 
            gr.update(visible=is_list),
            gr.update(value=judge_evaluation, visible=True)
        )
    
    # Function to generate PDF when button is clicked
    def on_download_click(response):
        pdf_path = create_pdf(response)
        if pdf_path:
            return pdf_path
        return None

    # Connect components
    submit_button.click(
        fn=chat_interface,
        inputs=[message_input, model_selector, use_csv_checkbox, use_pdf_checkbox, use_judge_checkbox],
        outputs=[response_output, is_list_state, judge_output]
    ).then(
        fn=update_ui,
        inputs=[response_output, is_list_state, judge_output],
        outputs=[response_output, is_list_state, pdf_row, judge_output]
    )
    
    download_button.click(
        fn=on_download_click,
        inputs=[response_output],
        outputs=[pdf_output]
    )

if __name__ == "__main__":
    demo.launch(share=True)
