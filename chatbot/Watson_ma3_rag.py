# Watson_agent.py

from decouple import config
from langchain_ibm import WatsonxLLM
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

# Watsonx konfiguration
WX_API_KEY = config("WX_API_KEY")
WX_PROJECT_ID = config("WX_PROJECT_ID")
WX_API_URL = "https://us-south.ml.cloud.ibm.com"  # eller det endpoint du bruger

# Initialiser Watsonx LLM
llm = WatsonxLLM(
    model_id="ibm/granite-3-8b-instruct",
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

def ask_watson(message: str) -> str:
    """Send a prompt to Watsonx LLM and return the response."""
    return llm.invoke(message)


if __name__ == "__main__":
    print(ask_watson("Hello Watson, how are you today?"))
