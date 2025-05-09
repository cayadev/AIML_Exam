

# main_agent.py

from decouple import config
from dotenv import load_dotenv
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ma3_for_inspo.src.llm import LLMCaller


load_dotenv()

# Step 1: Load credentials
WX_API_KEY = config("WX_API_KEY")
WX_PROJECT_ID = config("WX_PROJECT_ID")
WX_API_URL = "https://us-south.ml.cloud.ibm.com"

# Step 2: Set up LLM
model = LLMCaller(
    api_key=WX_API_KEY,
    project_id=WX_PROJECT_ID,
    api_url=WX_API_URL,
    model_id="watsonx/ibm/granite-3-8b-instruct",
    params={
        GenParams.TEMPERATURE: 0.6,
        GenParams.MAX_NEW_TOKENS: 300,
    }
)

# Step 3: Test agent
prompt = "Suggest best irrigation method for maize with 30°C and 70% humidity"
response = model.invoke(prompt)
print(response)
