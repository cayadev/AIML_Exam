# Watson_Chatbot.py

import gradio as gr
import pandas as pd
from Watson_Agent import ask_watson
from Watson_Style import WatsonDarkTheme  # Custom theme import

# Load company data
company_data = pd.read_csv("data.csv")

def chatbot_interface(user_input):
    # Check if user input matches anything in the data.csv file
    match_found = company_data.apply(
        lambda row: user_input.lower() in row.astype(str).str.lower().to_string(), axis=1
    ).any()

    data_match = "✅ Found something relevant in company data." if match_found else "ℹ️ No direct match found in data."

    # Get Watson response
    watson_response = ask_watson(user_input)

    return f"{data_match}\n\n💬 Watson says: {watson_response}"

# Build Gradio UI
demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(label="Ask a question", placeholder="e.g. What's the return policy?"),
    outputs=gr.Textbox(label="Response"),
    title="Watson-Powered Company Chatbot",
    description="Ask questions that combine IBM Watson AI and your company's data.",
    theme=WatsonDarkTheme()  # Apply the custom theme
)

if __name__ == "__main__":
    demo.launch()
