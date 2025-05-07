import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from the .env file
load_dotenv()

import time
import chatbot_utils as chatbot_utils
# import common.openai_tools.hni_hotel_room_lookup as hotel_room_lookup_tool
# import common.openai_tools.hni_ice_cream_sales_prediction_tool as ice_cream_sales_prediction_tool
# import common.openai_tools.hni_text_upload_tool as text_upload_tool
# import common.openai_tools.hni_weather_forecast_tool as weather_forecast_tool
import gradio as gr
import pandas as pd
import tiktoken
from chatbot.agent import Agent
from chatbot.chatbot_hni_style import Seafoam
from decouple import config
import os
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams

encoder = tiktoken.get_encoding("cl100k_base")
seafoam = Seafoam()
title = "ChatGPT Agent Demo"
COLLECTION_NAME = "hotel_info_collection"

all_available_function_names = [
    "Ice cream sales prediction",
    "Hotel room lookup",
    "Weather forecast",
]

# Get today's date
today = datetime.date.today()

# Get the weekday (Monday is 0 and Sunday is 6)
weekday = today.weekday()

SYSTEM_PROMPT = f"You are a helpful assistant for a successful hotel named 'Happy Hotel' that also has a successful ice cream store named 'Happy Scoops'. Your answers must be detailed and concise. You are speaking with the hotel manager. Todays date is: {today}, and the weekday is: {weekday}. You must not make up information without data to back it up."

# IBM Watson configuration
WX_API_KEY = os.getenv("WX_API_KEY")
WX_PROJECT_ID = config("WX_PROJECT_ID")
WX_API_URL = os.getenv("WX_API_URL", "https://us-south.ml.cloud.ibm.com")

def update_latest_used_data(state: gr.State) -> list:
    """
    Updates the list of latest used tables based on the state dictionary.

    Args:
        state (dict[str, list[Any]]): The state dictionary containing the latest tool outputs.

    Returns:
        list[gr.Dataframe]: The updated list of latest used tables as gr.Dataframe objects.
    """
    k = 0
    dataframes = []

    tool_outputs = state["tool_outputs"]

    for name, data in tool_outputs.items():
        if isinstance(data, pd.DataFrame):
            dataframes.append(gr.Dataframe(value=data, visible=True, label=name))

    k = len(dataframes)
    m = len(all_available_function_names)

    # Return the dataframes and fill the rest with empty dataframes. We add empty dataframes since gradio expects a constant number of elements (which is define in MAX_OUTPUTS)
    return dataframes + [gr.Dataframe(visible=False)] * (m - k)

def show_new_text_tool_creation() -> tuple[gr.Textbox, gr.File, gr.Button]:
    text_tool_description = gr.Textbox(
        "Describe the contents and/or purpose of the Text file",
        label="Text File Description",
        visible=True,
        interactive=True,
    )

    text_file_upload = gr.File(
        label="Text File Upload",
        visible=True,
        file_count="multiple",
    )

    create_new_text_tool_button = gr.Button(
        value="Create new text tool",
        size="sm",
        visible=False,
        interactive=True,
    )

    return text_tool_description, text_file_upload, create_new_text_tool_button

def upload_to_vdb(
    state: gr.State,
    uploaded_files: list,
    text_tool_description: str,
) -> gr.State:
    all_chunks = []

    qdrant_client = state["qdrant_client"]

    state["uploaded_file_names"] = [u_f.name for u_f in uploaded_files]
    state["uploaded_file_descriptions"] = text_tool_description

    for u_f in uploaded_files:
        chunks = chatbot_utils.read_and_split_text(
            file_path=u_f.name,
            max_token_size=200,
            overlap=50,
        )

        all_chunks += chunks

    ids = list(range(len(all_chunks)))

    qdrant_client.add(
        collection_name="text_file_upload_collection",
        documents=all_chunks,
        ids=ids,
    )

    state["qdrant_client"] = qdrant_client

    all_available_function_names = [
        "Ice cream sales prediction",
        "Hotel room lookup",
        "Weather forecast",
        "Additional Information",
    ]

    tools_checkboxes = gr.CheckboxGroup(
        all_available_function_names,
        label="Tools",
        value=all_available_function_names,
        interactive=True,
    )

    return state, tools_checkboxes

def init_qdrant_client(state: gr.State) -> gr.State:
    state["qdrant_client"] = QdrantClient(":memory:")
    state["tool_outputs"] = {}
    return state

def clear_chat(state: gr.State) -> tuple[list, gr.State]:
    state["tool_outputs"] = {}
    return [], state

def add_text(history: list, text: str) -> tuple[list, str]:
    history = history + [[text, None]]
    return history, ""

def generate_response(
    history: list,
    model_choice: str,
    state: gr.State,
    selected_tools: gr.CheckboxGroup,
) -> tuple[list, gr.State]:
    if not history:
        return [], state

    question = history[-1][0]
    memory = []

    for i, (user_msg, bot_msg) in enumerate(history[:-1]):
        memory.append({"role": "user", "content": user_msg})
        memory.append({"role": "assistant", "content": bot_msg})

    tools = []
    tool_name_dict = {}

    if "Ice cream sales prediction" in selected_tools:
        tools.append(ice_cream_sales_prediction_tool.IceCreamSalesPredictionTool)
        tool_name_dict[ice_cream_sales_prediction_tool.IceCreamSalesPredictionTool.__name__] = "Ice cream sales prediction"

    if "Hotel room lookup" in selected_tools:
        tools.append(hotel_room_lookup_tool.HotelRoomLookupTool)
        tool_name_dict[hotel_room_lookup_tool.HotelRoomLookupTool.__name__] = "Hotel room lookup"

    if "Weather forecast" in selected_tools:
        tools.append(weather_forecast_tool.WeatherForecastTool)
        tool_name_dict[weather_forecast_tool.WeatherForecastTool.__name__] = "Weather forecast"

    if "Additional Information" in selected_tools:
        tools.append(text_upload_tool.TextUploadTool)
        tool_name_dict[text_upload_tool.TextUploadTool.__name__] = "Additional Information"

    # Map model choice to actual model ID
    model_mapping = {
        "IBM Watson Granite 3-8B": "watsonx/ibm/granite-3-8b-instruct",
        "IBM Watson Granite 13B": "watsonx/ibm/granite-13b-instruct",
    }
    
    model_id = model_mapping.get(model_choice, "watsonx/ibm/granite-3-8b-instruct")
    
    agent = Agent(
        tools=tools,
        memory=memory,
        base_prompt=SYSTEM_PROMPT,
        tool_name_dict=tool_name_dict,
        model=model_id,
        improve_final_answer=True,
    )

    bot_response = ""
    for chunk in agent.generate_response(question, qdrant_client=state["qdrant_client"]):
        bot_response += chunk
        history[-1][1] = bot_response
        yield history, state

    state["tool_outputs"] = agent.latest_tool_outputs

    return history, state

with gr.Blocks(
    title=title,
    theme=seafoam,
    css=".gradio-container {background-color: lightgray}",
) as blk:
    state = gr.State({"qdrant_client": None, "tool_outputs": {}})

    with gr.Row():
        with gr.Column(scale=1):
            model_choice = gr.Radio(
                ["IBM Watson Granite 3-8B", "IBM Watson Granite 13B"],
                label="Model",
                value="IBM Watson Granite 3-8B",
            )

            tools_checkboxes = gr.CheckboxGroup(
                all_available_function_names,
                label="Tools",
                value=all_available_function_names,
                interactive=True,
            )

            text_tool_description, text_file_upload, create_new_text_tool_button = show_new_text_tool_creation()
            text_file_upload.change(upload_to_vdb, [state, text_file_upload, text_tool_description], [state, tools_checkboxes], queue=False)

            tool_outputs = []

            with gr.Group():
                for i in range(len(all_available_function_names)):
                    df = gr.Dataframe(visible=False)
                    tool_outputs.append(df)

        with gr.Column(scale=2):
            bot = gr.Chatbot(height=500, value=[[None,"Hi! What can I help you with today?"]])
            textbox = gr.Textbox(
                placeholder="Write here...",
                label="",
                interactive=True,
            )

            txt_msg = textbox.submit(add_text, [bot, textbox], [bot, textbox], queue=False).then(
                generate_response, [bot, model_choice, state, tools_checkboxes], [bot, state], api_name="bot_response",
            )

            txt_msg.then(lambda: gr.Textbox(interactive=True), None, [textbox], queue=False).then(update_latest_used_data, state, tool_outputs, queue=False)

            with gr.Row():
                clear_chat_button = gr.Button(
                    value="Clear conversation",
                    size="sm",
                    visible=True,
                    interactive=True,
                )

                ask_question_button = gr.Button(
                    value="Ask",
                    size="sm",
                    visible=True,
                    interactive=True,
                )

                clear_chat_button.click(clear_chat, [state], [bot, state], queue=False)
                ask_question_button.click(add_text, [bot, textbox], [bot, textbox], queue=False).then(
                    generate_response, [bot, model_choice, state, tools_checkboxes], [bot, state], api_name="bot_response",
                ).then(
                    lambda: gr.Textbox(interactive=True), None, [textbox], queue=False,
                ).then(update_latest_used_data, state, tool_outputs, queue=False)

    with gr.Tab("About the demo"):
        gr.Markdown(
            """
            # Hotel Agent Demo with IBM Watson

            This is a demo of a chatbot agent that can answer questions about a hotel and its services.

            PLEASE test the demo and get familiar with it before showing it to others.

            The agent can answer any question that the language model can normally answer, but it is also equipped with tools to answer questions within specific domains.

            The tools available are:
            1. Ice cream sales prediction - Predicts the sales of ice cream based on the weather forecast.
            2. Hotel room lookup - Looks up information about hotel rooms.
            3. Weather forecast - Provides the weather forecast for the next 14 days.

            By pressing the "Create new text tool" button, you can also upload one or more word files or pdf files which will be available for the Agent as a new tool. It is important that you give a description of the contents and/or purpose of the text file.

            You can toggle the tools you want to use by checking the checkboxes in the "Tools" section.

            Here are some example questions you can ask the agent:
            - "What is the weather forecast for tomorrow?"
            - "What is the price for a single room?"
            - "What is the sales prediction for ice cream today?"

            You can find all the data used in this demo in the files field below. It is recommended to download the file "HappyHotelInfo.docx" and upload it as a tool during demo to show how we can add new knowledge to the agent.
            """,
        )

        gr.File(["./src/demo_data/HappyHotelInfo.docx", "./src/demo_data/HotelRoomsDataset.xlsx", "./src/demo_data/IceCreamSales.csv"], label="Files")

    blk.load(init_qdrant_client, inputs=[state], outputs=[state])

if __name__ == "__main__":
    blk.queue().launch(server_name="0.0.0.0", server_port=8182)