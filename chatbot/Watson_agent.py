# Watson_Agent.py

import json
from ibm_watson import AssistantV2
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
from env import WATSON_API_KEY, WATSON_ASSISTANT_ID, WATSON_URL

# Initialize Watson Assistant
authenticator = IAMAuthenticator(WATSON_API_KEY)
assistant = AssistantV2(
    version='2021-06-14',
    authenticator=authenticator
)
assistant.set_service_url(WATSON_URL)

# Create session
session_response = assistant.create_session(
    assistant_id=WATSON_ASSISTANT_ID
).get_result()
session_id = session_response['session_id']


def ask_watson(message: str) -> str:
    """Send a message to Watson Assistant and return the response."""
    response = assistant.message(
        assistant_id=WATSON_ASSISTANT_ID,
        session_id=session_id,
        input={'message_type': 'text', 'text': message}
    ).get_result()

    try:
        return response['output']['generic'][0]['text']
    except (KeyError, IndexError):
        return "Sorry, I didn't understand that."
