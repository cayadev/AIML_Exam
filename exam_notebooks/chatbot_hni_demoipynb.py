import datetime
from dotenv import load_dotenv
from qdrant_client import QdrantClient

# Load environment variables from the .env file
load_dotenv()

import time
import chatbot_utils as chatbot_utils
import common.openai_tools.hni_hotel_room_lookup as hotel_room_lookup_tool
import common.openai_tools.hni_ice_cream_sales_prediction_tool as ice_cream_sales_prediction_tool
import common.openai_tools.hni_text_upload_tool as text_upload_tool
import common.openai_tools.hni_weather_forecast_tool as weather_forecast_tool
import gradio as gr
import pandas as pd
import tiktoken
from chatbot.agent import Agent
from chatbot.chatbot_hni_style import Seafoam

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
        info="Tools to use",
    )

    return state, tools_checkboxes

def init_qdrant_client(state: gr.State) -> gr.State:
    if state["qdrant_client"] is None:
        print("Initializing Qdrant client")
        q_client = QdrantClient(":memory:")
        state["qdrant_client"] = q_client

    return state

def clear_chat(state: gr.State) -> tuple[list, dict]:
    state["conversation_memory"] = []

    return [], state

def add_text(history: list, text: str) -> tuple[list, gr.Textbox]:
    history = [*history, (text, None)]
    return history, gr.Textbox(value="", interactive=False)

def generate_response(
    history: list,
    model_choice: str,
    state: gr.State,
    selected_tools: gr.CheckboxGroup,
) -> tuple[list, gr.State]:
    conversation_memory = state["conversation_memory"]

    all_available_functions = {
        "Ice cream sales prediction": ice_cream_sales_prediction_tool.get_ice_cream_sales_prediction_tool(),
        "Hotel room lookup": hotel_room_lookup_tool.get_hotel_room_lookup_tool(),
        "Weather forecast": weather_forecast_tool.get_weather_forecast_tool(),
    }

    if "Additional Information" in selected_tools:
        uploaded_file_names = state["uploaded_file_names"]
        uploaded_file_descriptions = state["uploaded_file_descriptions"]
        all_available_functions["Additional Information"] = text_upload_tool.get_text_search_tool(file_names=uploaded_file_names, file_descriptions=uploaded_file_descriptions)

    tools = []
    tools_name_dict = {}
    tools_text = ""

    for func_name, cls in all_available_functions.items():
        if func_name in selected_tools:
            tools.append(cls)
            tools_text += f"{func_name}\n"
            tools_name_dict[cls.__name__] = func_name

    if tools_text == "":
        tools_text = "None"

    sp = SYSTEM_PROMPT.replace("<TOOLS>", tools_text)

    agent = Agent(
        tools=tools,
        tool_name_dict=tools_name_dict,
        memory=conversation_memory,
        base_prompt=sp,
        model=model_choice,
        function_call="auto",
        temperature=0.0,
        improve_final_answer=True,
    )

    response = agent.generate_response(question=history[-1][0], qdrant_client=state["qdrant_client"])

    history[-1][1] = ""

    for r in response:
        history[-1][1] += r
        yield history, state
        time.sleep(0.01)

    state["conversation_memory"] = agent.memory
    if agent._tool_outputs:
        state["tool_outputs"] = agent._tool_outputs

    return history, state

with gr.Blocks(theme=seafoam) as blk:
    state = gr.State(value={"conversation_memory": [], "qdrant_client": None, "tool_outputs": {}})

    gr.Image("./src/twoday-wordmark-RGB_BLACK.png", label="Logo", interactive=False, container=False, width=200, show_download_button=False)

    with gr.Tab("Hotel Agent") as tab1, gr.Row() as r1:
        with gr.Column(scale=1):
            model_choice = gr.Dropdown(["gpt-35-turbo", "gpt4"], label="Model", value="gpt-35-turbo")
            tools_checkboxes = gr.CheckboxGroup(
                all_available_function_names,
                label="Tools",
                info="Tools to use",
            )
            create_new_text_tool_button = gr.Button(
                value="Create new text tool",
                size="sm",
                visible=True,
                interactive=True,
            )

            text_tool_description = gr.Textbox(
                "Describe the contents and/or purpose of the Text file",
                label="Text File Description",
                visible=False,
                interactive=True,
            )

            text_file_upload = gr.File(
                label="Text File Upload (Only word files and pdf files are supported)",
                visible=False,
                file_types=["docx", "pdf"],
                file_count="multiple",
            )

            create_new_text_tool_button.click(show_new_text_tool_creation, None, [text_tool_description, text_file_upload, create_new_text_tool_button], queue=False)
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
            # ChatGPT Hotel Agent Demo

            This is a demo of a chatbot agent that can answer questions about a hotel and its services.

            PLEASE test the demo and get familiar with it before showing it to others.

            The agent can answer any question that ChatGPT can normally answer, but it is also equipped with tools to answer questions within specific domains.

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