from Watson_chatbot_backend import ask_watson

# Basic usage
response = ask_watson("What are the growing conditions for cotton?")

# Customized usage
response = ask_watson(
    message="Tell me about rice farming",
    model_id="ibm/granite-3-8b-instruct",  # Choose your model
    use_csv=True,                          # Use CSV data
    use_pdf=True                           # Use PDF data
)