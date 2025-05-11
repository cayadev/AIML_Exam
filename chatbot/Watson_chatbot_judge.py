import os
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from typing import Dict, Any
import re

# Global variables to store initialized components
watson_model = None

def initialize_watson_judge() -> WatsonxLLM:
    """Initialize the Watson model for judging responses using Watsonx.
    
    Returns:
        The initialized Watson model
    """
    global watson_model
    
    if watson_model is not None:
        return watson_model
    
    # Watsonx configuration - using os.environ
    WX_API_KEY = os.environ.get("WX_API_KEY")
    WX_PROJECT_ID = os.environ.get("WX_PROJECT_ID")
    WX_API_URL = "https://us-south.ml.cloud.ibm.com"  # or the endpoint you use
    
    # Initialize Watsonx LLM with the fixed model
    model_id = "meta-llama/llama-3-3-70b-instruct"
    
    # Initialize Watsonx LLM
    watson_model = WatsonxLLM(
        model_id=model_id,
        url=WX_API_URL,
        apikey=WX_API_KEY,
        project_id=WX_PROJECT_ID,
        params={
            GenParams.DECODING_METHOD: "greedy",
            GenParams.TEMPERATURE: 0.1,  # Low temperature for more deterministic outputs
            GenParams.MIN_NEW_TOKENS: 5,
            GenParams.MAX_NEW_TOKENS: 512,
            GenParams.REPETITION_PENALTY: 1.2,
        }
    )
    
    return watson_model

def create_judge_prompt(user_query: str, chatbot_response: str) -> str:
    """Create a prompt for the judge to evaluate the chatbot's response.
    
    Args:
        user_query: The original user question
        chatbot_response: The response from the chatbot
        
    Returns:
        A formatted prompt for the judge
    """
    prompt = f"""You are an expert judge evaluating the quality of an AI assistant's response to a user query about agriculture. 
Please assess the following response based on these criteria:

1. Factual Correctness: Is the information accurate and supported by relevant data?
2. Relevance: Is the response contextually appropriate to the user's query?
3. Clarity and Fluency: Is the response coherent, well-structured, and easy to understand?
4. Helpfulness: Does the output support practical decision-making for farmers?

User Query: {user_query}

Assistant's Response: {chatbot_response}

Based on the above criteria, provide a single overall score from 1 to 5, where:
1: Poor quality response with significant issues
2: Below average response with several limitations
3: Average response that meets basic needs
4: Good response that meets most needs
5: Excellent response that fully addresses the query

Format your response as follows:

Overall Score: [score] - [brief explanation of the score, mentioning how the response performed on the four criteria]
"""
    return prompt

def judge_response(user_query: str, chatbot_response: str) -> Dict[str, Any]:
    """Judge the quality of a chatbot response using the Watson model.
    
    Args:
        user_query: The original user question
        chatbot_response: The response from the chatbot
        
    Returns:
        Dictionary containing the evaluation results
    """
    try:
        # Initialize the Watson model
        wx = initialize_watson_judge()
        
        # Create the prompt for evaluation
        prompt = create_judge_prompt(user_query, chatbot_response)
        
        # Get the evaluation from the Watson model
        evaluation = wx.invoke(prompt)
        
        # Parse the evaluation results
        results = parse_evaluation_results(evaluation)
        
        return {
            "raw_evaluation": evaluation,
            "parsed_results": results,
            "success": True
        }
    except Exception as e:
        return {
            "raw_evaluation": f"Error evaluating response: {str(e)}",
            "parsed_results": {
                "overall_score": 0,
                "explanation": "Failed to evaluate the response due to an error."
            },
            "success": False
        }

def parse_evaluation_results(evaluation: str) -> Dict[str, Any]:
    """Parse the evaluation results from the Watson model.
    
    Args:
        evaluation: The raw evaluation text from the Watson model
        
    Returns:
        Dictionary containing the parsed evaluation results
    """
    # Initialize default values
    results = {
        "overall_score": 0,
        "explanation": ""
    }
    
    # Extract overall score
    if "Overall Score:" in evaluation:
        score_part = evaluation.split("Overall Score:")[1].strip()
        try:
            # Try to extract the score
            score_match = re.search(r'(\d+(?:\.\d+)?)', score_part)
            if score_match:
                score = float(score_match.group(1))
                results["overall_score"] = score
            
            # Extract the explanation (everything after the score)
            explanation_match = re.search(r'\d+(?:\.\d+)?\s*-\s*(.*)', score_part)
            if explanation_match:
                explanation = explanation_match.group(1).strip()
                results["explanation"] = explanation
            else:
                results["explanation"] = score_part
        except (IndexError, ValueError) as e:
            results["explanation"] = f"Error parsing score: {str(e)}"
    
    return results

def format_judge_result(judge_result: Dict[str, Any]) -> str:
    """Format the judge results for display.
    
    Args:
        judge_result: The judge evaluation results
        
    Returns:
        Formatted string for display
    """
    if not judge_result.get("success", False):
        return "⚠️ Judge evaluation failed. Please check the model configuration."
    
    results = judge_result["parsed_results"]
    score = results["overall_score"]
    explanation = results["explanation"]
    
    # Determine emoji based on score
    if score >= 4:
        emoji = "🌟"
    elif score >= 3:
        emoji = "⭐"
    elif score >= 2:
        emoji = "✅"
    else:
        emoji = "⚠️"
    
    formatted_result = f"## {emoji} Response Evaluation\n\n"
    formatted_result += f"**Overall Score**: {score:.1f}/5\n\n"
    formatted_result += f"{explanation}"
    
    return formatted_result

if __name__ == "__main__":
    # Test the judge functionality
    test_query = "What are the best soil conditions for growing wheat?"
    test_response = """
    Wheat grows best in well-drained loamy soils with a pH between 6.0 and 7.0. 
    The soil should have good water-holding capacity but not be waterlogged. 
    Wheat requires moderate to high levels of nitrogen, phosphorus, and potassium.
    Sandy soils may require more frequent irrigation and fertilization.
    Clay soils may need better drainage to prevent root diseases.
    """
    
    # Test with the Watson model
    result = judge_response(test_query, test_response)
    
    if result["success"]:
        print(format_judge_result(result))
    else:
        print("Judge evaluation failed:", result["raw_evaluation"])