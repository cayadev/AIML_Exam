# Watson_Chatbot.py

import gradio as gr
import pandas as pd
from Watson_Agent import ask_watson

# Load company data
company_data = pd.read_csv("data.csv")

def chatbot_interface(user_input):
    # Simple search in CSV to add context (very basic for now)
    if any(company_data.apply(lambda row: user_input.lower() in row.astype(str).str.lower().to_string(), axis=1)):
        data_match = "I found something relevant in our company data."
    else:
        data_match = "No direct match found in our data."

    # Ask Watson
    watson_response = ask_watson(user_input)

    return f"{data_match}\n\nWatson says: {watson_response}"

# Gradio UI
demo = gr.Interface(fn=chatbot_interface,
                    inputs=gr.Textbox(label="Ask something"),
                    outputs="text",
                    title="Company + Watson Chatbot")

if __name__ == "__main__":
    demo.launch()
