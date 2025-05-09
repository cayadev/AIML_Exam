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
                GenParams.MAX_NEW_TOKENS: 1000,
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
        
        # Define path to data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize vector database
        pdf_data_path = os.path.join(current_dir, "data")
        vector_db = initialize_vector_db(pdf_data_path, embeddings, current_dir)
    
    # Load crop descriptions if not already done
    if crop_descriptions is None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_path = os.path.join(current_dir, "data", "Crop_Conditions_Dataset.csv")
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
    
    # Initialize if not already done
    if model_id not in llm_models or (use_pdf and vector_db is None) or (use_csv and crop_descriptions is None):
        llm, vector_db, crop_descriptions = initialize_watson(model_id)
    else:
        llm = llm_models[model_id]
    
    # Check if the question is about crops or farming
    crop_keywords = ["crop", "farm", "soil", "weather", "agriculture", "plant", "harvest", 
                    "disease", "yield", "moisture", "pH", "temperature", "rainfall", 
                    "humidity", "irrigation", "fertilizer", "pesticide", "cotton", "wheat",
                    "rice", "maize", "soybean"]
    
    is_crop_question = any(keyword.lower() in message.lower() for keyword in crop_keywords)
    
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
            
            # Create a prompt with retrieved documents as context
            prompt = f"""You are an agricultural assistant that helps farmers with crop conditions information.
Use the following information from agricultural documents to answer the question."""

            if use_csv and crop_descriptions:
                prompt += f"""If you don't know the answer or the information is not in the documents, use the crop condition descriptions provided.

Retrieved Documents:
{docs_content}

Crop Condition Descriptions:
{chr(10).join(crop_descriptions)}"""
            else:
                prompt += f"""

Retrieved Documents:
{docs_content}"""
            
            prompt += f"""

Question: {message}

Answer:"""
            
            # Get response from LLM
            return llm.invoke(prompt)
        
        # Fall back to using crop descriptions if no relevant documents found or not using PDFs
        elif use_csv and crop_descriptions:
            prompt = f"""You are an agricultural assistant that helps farmers with crop conditions information.
Use the following crop condition descriptions to answer the question. If you don't know the answer or the information is not in the descriptions, just say so.

Crop Condition Descriptions:
{chr(10).join(crop_descriptions)}

Question: {message}

Answer:"""
            
            # Get response from LLM
            return llm.invoke(prompt)
        else:
            # No data sources available, use the LLM directly
            return llm.invoke(f"You are an agricultural assistant. Answer this question: {message}")
    else:
        # For non-crop questions, use the LLM directly
        return llm.invoke(message)

def get_available_models() -> List[str]:
    """Return a list of available Watson models that can be used.
    
    Returns:
        List of model IDs as strings
    """
    # This is a simplified list - in a real application, you might query the Watson API
    return [
        "ibm/granite-3-8b-instruct",
        "ibm/granite-13b-instruct",
        "ibm/granite-20b-instruct"
    ]

if __name__ == "__main__":
    # Initialize Watson components when run directly
    llm, vector_db, crop_descriptions = initialize_watson()
    
    # Test with different configurations
    print("Using both CSV and PDF data:")
    print(ask_watson("What are the soil conditions for wheat crops in North India?", 
                    use_csv=True, use_pdf=True))
    
    print("\nUsing only CSV data:")
    print(ask_watson("What are the soil conditions for cotton crops?", 
                    use_csv=True, use_pdf=False))
    
    print("\nUsing only PDF data:")
    print(ask_watson("Tell me about soybean farming.", 
                    use_csv=False, use_pdf=True))
    
    print("\nUsing no additional data:")
    print(ask_watson("Hello Watson, how are you today?", 
                    use_csv=False, use_pdf=False))
