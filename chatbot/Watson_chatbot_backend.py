# Watson_chatbot_backend.py

import os
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
import pandas as pd

# New imports for RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ibm import WatsonxEmbeddings
from langchain_chroma import Chroma
from typing import Optional, Dict, Any, List, Tuple
import re

# Global variables to store initialized components
llm_models = {}
vector_db = None
crop_descriptions = None

def initialize_watson(model_id: str = "ibm/granite-3-8b-instruct"):
    """Initialize the Watson LLM and embeddings models.
    
    Args:
        model_id: The Watson model ID to use
        
    Returns:
        Tuple containing the LLM, vector_db, and crop_descriptions
    """
    global llm_models, vector_db, crop_descriptions
    
    # Watsonx configuration - using os.environ
    WX_API_KEY = os.environ.get("WX_API_KEY")
    WX_PROJECT_ID = os.environ.get("WX_PROJECT_ID")
    WX_API_URL = "https://us-south.ml.cloud.ibm.com"  # or the endpoint you use

    # Initialize Watsonx LLM if not already created for this model
    if model_id not in llm_models:
        llm_models[model_id] = WatsonxLLM(
            model_id=model_id,
            url=WX_API_URL,
            apikey=WX_API_KEY,
            project_id=WX_PROJECT_ID,
            params={
                GenParams.DECODING_METHOD: "greedy",
                GenParams.TEMPERATURE: 0.4,
                GenParams.MIN_NEW_TOKENS: 5,
                GenParams.MAX_NEW_TOKENS: 2500,  # Increased from 1000 to 2500 for longer responses
                GenParams.REPETITION_PENALTY: 1.2,
            }
        )
    
    # Get the requested LLM
    llm = llm_models[model_id]

    # Initialize Watsonx Embeddings for RAG (if not already done)
    if vector_db is None:
        embeddings = WatsonxEmbeddings(
            model_id="ibm/granite-embedding-278m-multilingual",
            url=WX_API_URL,
            apikey=WX_API_KEY,
            project_id=WX_PROJECT_ID
        )
        
        # Define paths to data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level to project root
        
        # Initialize vector database
        pdf_data_path = os.path.join(project_root, "data")
        vector_db = initialize_vector_db(pdf_data_path, embeddings, project_root)
    
    # Load crop descriptions if not already done
    if crop_descriptions is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)  # Go up one level to project root
        data_path = os.path.join(project_root, "data", "Crop_Conditions_Dataset.csv")
        crop_descriptions = load_crop_data(data_path)
    
    return llm, vector_db, crop_descriptions

def load_crop_data(data_path):
    """Load the crop conditions dataset and extract descriptions."""
    df = pd.read_csv(data_path)
    # Return only the description column as a list
    return df['crop_condition_description'].tolist()

# Function to load PDF files and create document chunks
def load_pdf_documents(pdf_data_path):
    """Load PDF files from the data directory and split them into chunks."""
    pdf_files = [f for f in os.listdir(pdf_data_path) if f.endswith('.pdf')]
    documents = []
    
    for pdf_file in pdf_files:
        file_path = os.path.join(pdf_data_path, pdf_file)
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
        
    # Add source information to metadata
    for doc in documents:
        doc.metadata["source"] = doc.metadata.get("source", "").split("/")[-1]
    
    return documents

# Function to split documents into smaller chunks for better retrieval
def split_documents(documents):
    """Split documents into smaller chunks for better retrieval."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    
    return text_splitter.split_documents(documents)

# Initialize vector database for document retrieval
def initialize_vector_db(pdf_data_path, embeddings, current_dir):
    """Initialize the vector database with document chunks."""
    # Load and process PDF documents
    documents = load_pdf_documents(pdf_data_path)
    chunks = split_documents(documents)
    
    # Create vector database
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=os.path.join(current_dir, "chroma_db")
    )
    
    return vector_db

def ask_watson(
    message: str, 
    model_id: str = "ibm/granite-3-8b-instruct",
    use_csv: bool = True,
    use_pdf: bool = True
) -> str:
    """Send a prompt to Watsonx LLM and return the response.
    
    Args:
        message: The user's question or message
        model_id: The Watson model ID to use
        use_csv: Whether to include CSV crop data
        use_pdf: Whether to include PDF data
        
    Returns:
        str: The response from Watson
    """
    global llm_models, vector_db, crop_descriptions
    
    # Initialize Watson components if not already done
    if model_id not in llm_models or vector_db is None or crop_descriptions is None:
        llm, vector_db, crop_descriptions = initialize_watson(model_id)
    else:
        llm = llm_models[model_id]
    
    # Define base prompt
    base_prompt = "You are an agricultural assistant that helps farmers with crop conditions information."
    
    # Define list formatting instructions to avoid repetition
    list_format_instructions = """
Since this is a request for a priority list, action plan, or to-do list, format your response in a clear, structured way:
1. Start with a brief introduction
2. Present the items as a numbered list with clear titles for each item
3. For each item, provide a short description or explanation
4. End with a brief conclusion"""
    
    # Check if the question is about priority lists, action plans, or to-dos
    list_keywords = ["priority list", "priorities", "action plan", "to-do", "todo", "to do", 
                     "task list", "checklist", "steps", "action items", "roadmap", "milestones"]
    
    is_list_request = any(keyword.lower() in message.lower() for keyword in list_keywords)
    
    # Keywords for crop-related questions
    crop_keywords = ["crop", "farm", "agriculture", "soil", "plant", "harvest", "grow", 
                    "disease", "yield", "moisture", "pH", "temperature", "rainfall", 
                    "humidity", "irrigation", "fertilizer", "pesticide", "cotton", "wheat",
                    "rice", "maize", "soybean"]
    
    is_crop_question = any(keyword.lower() in message.lower() for keyword in crop_keywords)
    
    # Build the prompt based on the type of question
    prompt = base_prompt
    
    # First check if this is a question that might benefit from RAG with PDF documents
    if is_crop_question:
        retrieved_docs = []
        
        # Retrieve relevant documents from vector database if using PDFs
        if use_pdf and vector_db is not None:
            retrieved_docs = vector_db.similarity_search(message, k=3)
        
        # If we found relevant documents, use them for RAG
        if retrieved_docs:
            # Extract content from retrieved documents
            docs_content = "\n\n".join(f"Document {i+1} (from {doc.metadata.get('source', 'unknown')}): \n{doc.page_content}" 
                                    for i, doc in enumerate(retrieved_docs))
            
            # Add context to the prompt
            prompt += "\nUse the following information from agricultural documents to answer the question."

            if use_csv and crop_descriptions:
                prompt += f"""
If you don't know the answer or the information is not in the documents, use the crop condition descriptions provided.

Retrieved Documents:
{docs_content}

Crop Condition Descriptions:
{chr(10).join(crop_descriptions)}"""
            else:
                prompt += f"""

Retrieved Documents:
{docs_content}"""
        
        # Fall back to using crop descriptions if no relevant documents found or not using PDFs
        elif use_csv and crop_descriptions:
            prompt += f"""
Use the following crop condition descriptions to answer the question. If you don't know the answer or the information is not in the descriptions, just say so.

Crop Condition Descriptions:
{chr(10).join(crop_descriptions)}"""
    else:
        # For non-crop questions, use a simpler prompt
        prompt = "Answer this question:"
    
    # Add list formatting instructions if this is a list request
    if is_list_request:
        prompt += f"""

{list_format_instructions}"""
    
    # Add the user's question to the prompt
    prompt += f"""

Question: {message}

Answer:"""
    
    # Get response from LLM
    response = llm.invoke(prompt)
    
    # If this was a list request, ensure proper formatting
    if is_list_request:
        response = format_list_response(response)
        
    return response

def format_list_response(response: str) -> str:
    """Format a list-type response to ensure consistent structure.
    
    Args:
        response: The raw response from the LLM
        
    Returns:
        str: Formatted response with consistent structure
    """
    # Check if the response already has numbered items
    has_numbered_items = bool(re.search(r'^\s*\d+\.', response, re.MULTILINE))
    
    if not has_numbered_items:
        # Try to identify list items and format them
        lines = response.split('\n')
        formatted_lines = []
        item_number = 1
        
        for line in lines:
            # Look for potential list items (lines with keywords or bullet points)
            if re.match(r'^\s*[-•*]\s+', line) or any(keyword in line.lower() for keyword in ['step', 'task', 'action', 'priority']):
                # Replace bullet with number
                line = re.sub(r'^\s*[-•*]\s+', f"{item_number}. ", line)
                if not re.match(r'^\d+\.', line):
                    line = f"{item_number}. {line}"
                item_number += 1
            formatted_lines.append(line)
        
        response = '\n'.join(formatted_lines)
    
    # Add a special marker to indicate this is a formatted list
    return f"<FORMATTED_LIST>\n{response}\n</FORMATTED_LIST>"

def get_available_models() -> List[str]:
    """Return a list of available Watson models that can be used.
    
    Returns:
        List of model IDs as strings
    """
    # This is a simplified list - in a real application, you might query the Watson API
    return [
        "ibm/granite-3-8b-instruct",
        "meta-llama/llama-3-3-70b-instruct",
        "mistralai/mistral-large"
    ]

def generate_pdf_content(response: str) -> str:
    """Generate formatted content for PDF export.
    
    Args:
        response: The response from the LLM
        
    Returns:
        str: HTML-formatted content for PDF export
    """
    # Check if this is a formatted list response
    if "<FORMATTED_LIST>" in response and "</FORMATTED_LIST>" in response:
        # Extract the content between the tags
        content = response.split("<FORMATTED_LIST>")[1].split("</FORMATTED_LIST>")[0].strip()
        
        # Convert the content to HTML format
        lines = content.split('\n')
        html_content = "<h1>Generated Action Plan</h1>\n"
        
        in_list = False
        for line in lines:
            # Check if this line is a numbered item
            if re.match(r'^\s*\d+\.', line):
                if not in_list:
                    html_content += "<ol>\n"
                    in_list = True
                # Extract the item title if it exists (e.g., "1. Title: Description" -> "Title")
                title_match = re.match(r'^\s*\d+\.\s*([^:]+):(.*)', line)
                if title_match:
                    title, description = title_match.groups()
                    html_content += f"<li><strong>{title.strip()}</strong>{description}</li>\n"
                else:
                    html_content += f"<li>{line.split('.', 1)[1].strip()}</li>\n"
            else:
                if in_list:
                    html_content += "</ol>\n"
                    in_list = False
                html_content += f"<p>{line}</p>\n"
        
        if in_list:
            html_content += "</ol>\n"
        
        return html_content
    else:
        # For non-list responses, just wrap in paragraphs
        paragraphs = response.split('\n\n')
        html_content = "<h1>Generated Content</h1>\n"
        for para in paragraphs:
            if para.strip():
                html_content += f"<p>{para}</p>\n"
        return html_content

if __name__ == "__main__":
    # Initialize Watson components when run directly
    llm, vector_db, crop_descriptions = initialize_watson()
