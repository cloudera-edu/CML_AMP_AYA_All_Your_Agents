# ============================================================
# SQLite workaround for CML / Chroma
# MUST be before importing CrewAI or anything that imports Chroma
# ============================================================
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3


# ============================================================
# Imports
# ============================================================
from crewai import Agent, Task, Crew, Process, LLM
import asyncio
from textwrap import dedent
import os
import litellm
import uuid

litellm.set_verbose = False


# ============================================================
# Logging
# ============================================================
from pathlib import Path

log_file = "/tmp/crew.log"
Path(log_file).touch()


# ============================================================
# Bedrock LLM
# ============================================================
bedrock_model = os.getenv("AWS_BEDROCK_MODEL")

print("==============================================")
print("AWS_BEDROCK_MODEL:", bedrock_model)
print("==============================================")

if not bedrock_model:
    raise RuntimeError(
        "AWS_BEDROCK_MODEL environment variable is not set."
    )

# IMPORTANT:
# Use bedrock/<model>
# Do NOT use bedrock/converse/<model>
llm = LLM(
    model="bedrock/" + bedrock_model
)


# ============================================================
# Tools
# ============================================================
print("Importing available tools")

import tools.tool_ticket_query as tool_ticket_query

ticket_search_tool = tool_ticket_query.TicketListingTool()


import tools.tool_customer_consumption as tool_customer_consumption

consumption_change_tool = (
    tool_customer_consumption.ConsumptionMetricsTool()
)


import tools.tool_accounts_kg as knowledge_graph_accounts

account_info_tool = (
    knowledge_graph_accounts.ToolAccountsKnowledgeGraph()
)


import tools.tool_case_summarizer as tool_case_summarizer

case_summarizer_tool = (
    tool_case_summarizer.ToolCaseSummarizer()
)


# ============================================================
# Agent
# ============================================================
agent_1 = Agent(
    role=dedent(
        """
        Customer information lookup assistant.
        """
    ),

    backstory=dedent(
        """
        You are an advanced Cloudera Customer Enablement assistant,
        attempting to provide information about CML Customers,
        their deployed workspaces and entitlements, and usage
        information within those workspaces.

        This may include lookups of filed support cases,
        summaries of support cases, overall account consumption,
        workload consumption trends, and field information.
        """
    ),

    goal=dedent(
        """
        Perform the lookup task assigned to you by appropriate
        lookup tools and explain. Try to keep final answers
        in markdown format.
        """
    ),

    tools=[
        ticket_search_tool,
        case_summarizer_tool,
        consumption_change_tool,
        account_info_tool
    ],

    allow_delegation=False,
    max_iter=1,
    max_retry_limit=3,
    max_retries=3,
    llm=llm,
    verbose=True
)


# ============================================================
# Task
# ============================================================
import datetime

print("Defining Primary Task")

task_1 = Task(
    description=dedent(
        """
        Attempt to answer the user question about the customer.

        ---
        Request ID: "{req_id}"
        User Input: "{req_input}"
        """
    ),

    expected_output=dedent(
        """
        Output should be a well formatted list or statement
        with the results of the user request.
        """
    ),

    agent=agent_1,
)


# ============================================================
# Crew
# ============================================================
def crew_launch(req_id, req_input):

    print("Instantiating Crew")

    crew = Crew(
        agents=[agent_1],
        tasks=[task_1],
        verbose=True,
        process=Process.sequential
    )

    print("Setting req_input")

    inputs = {
        "req_id": req_id,
        "req_input": req_input
    }

    print("Kicking off crew")

    result = crew.kickoff(inputs=inputs)

    print(result.tasks_output)

    return result


# ============================================================
# Read logs
# ============================================================
def read_logs():

    with open(log_file, "r") as f:
        return f.read()


# ============================================================
# Gradio UI
# ============================================================
example = ""

print("Loading UI")

import gradio as gr


bot_msg = """
<strong style="text-align:left;">A.Y.A.</strong> - %s

%s
"""

human_msg = """
<p align="right">%s - <strong>User</strong></p>

%s
"""


startup_history = [
    (
        None,
        bot_msg % (
            datetime.datetime.now().strftime('%H:%M'),
            "Hello, how can I help you today?"
        )
    )
]


# ============================================================
# User message
# ============================================================
def display_user_message(message, chat_history):

    request_id = str(uuid.uuid4())[:8]

    message_text = message["text"]

    chat_history.append(
        (
            human_msg % (
                datetime.datetime.now().strftime('%H:%M'),
                message_text
            ),
            None
        )
    )

    return request_id, message_text, chat_history


# ============================================================
# Thinking message
# ============================================================
def display_thinking(chat_history):

    thinking_msg = """
    <h3 style="text-align:left;">🔄 Thinking...</h3>
    """

    chat_history.append(
        (
            None,
            thinking_msg
        )
    )

    return chat_history


# ============================================================
# Respond
# ============================================================
def respond(
    request_id,
    message_text,
    chat_history
):

    agent_usage_template = """
    <h3 style="text-align:left;">🛠️ Calling Agent ...</h3>

    `%s`
    """

    crew_response = crew_launch(
        request_id,
        message_text
    )

    try:

        tool_usage_file = open(
            "/tmp/%s" % request_id,
            "r"
        )

        tool_usage_text = tool_usage_file.read()

    except:

        tool_usage_text = "Unknown Agent"

    agent_usage_message = (
        agent_usage_template % tool_usage_text
    )

    chat_history.append(
        (
            None,
            agent_usage_message
        )
    )

    chat_history.append(
        (
            None,
            bot_msg % (
                datetime.datetime.now().strftime('%H:%M'),
                str(crew_response)
            )
        )
    )

    return chat_history


# ============================================================
# Status
# ============================================================
def maybe_update_status(
    status,
    chat_history
):

    crew_status = status

    chat_history.append(
        (
            None,
            bot_msg % crew_status
        )
    )


# ============================================================
# CSS
# ============================================================
css = """
footer{
    display:none !important
}

#examples_table {
    zoom: 70% !important;
}

#chatbot {
    flex-grow: 1 !important;
    overflow: auto !important;
}

#col {
    height: 75vh !important;
}

.info_md .container {
    border:1px solid #ccc;
    border-radius:5px;
    min-height:300px;
    color: #666;
    padding: 10px;
    background-color: whitesmoke;
}
"""


# ============================================================
# Header
# ============================================================
header_text = """
# All Your Agents

Meet **A.Y.A**, an Agentic Workflow Orchestrator
which deciphers and sends User requests to domain
specific AI Agents and Tools.
"""


# ============================================================
# Information
# ============================================================
info_text = """
<div class='container'>

## Agents A.Y.A. Can Use

**📝 Ticket Database Agent**

Queries a db for recent customer incident tickets.

##### 📖 Support Case Summarization Agent

AI Summarization Agent for support case details and comments.

##### 📈 Consumption Metrics Agent

Looks up recent trends in Customer Consumption.

##### 👤 Account Personnel Registry Agent

Knowledge graph lookup of Customer Account Team personnel.

</div>
"""


# ============================================================
# Gradio Theme
# ============================================================
theme = gr.themes.Base().set(
    body_background_fill=
    "url('file=/home/cdsw/assets/background.png') "
    "#FFFFFF no-repeat center bottom / "
    "100svw auto padding-box fixed"
)


# ============================================================
# Input
# ============================================================
input = gr.MultimodalTextbox(
    scale=5,
    show_label=False,
    file_types=["text"]
)


# ============================================================
# Gradio Application
# ============================================================
with gr.Blocks(
    css=css,
    theme=theme,
    title="A.Y.A."
) as demo:

    request_id = gr.State("")

    request_text = gr.State("")


    with gr.Row():

        gr.Markdown(header_text)


    with gr.Row():

        # ----------------------------------------------------
        # Left column
        # ----------------------------------------------------
        with gr.Column(scale=6):

            info = gr.Markdown(
                info_text,
                elem_classes=["info_md"]
            )

            example_num = gr.Textbox(
                visible=False
            )

            with gr.Accordion(
                "Example User Inputs",
                open=True
            ):

                examples_2 = gr.Examples(
                    [
                        [
                            1,
                            {
                                "text":
                                "Show me the weekly average "
                                "consumption changes for customers"
                            }
                        ],

                        [
                            2,
                            {
                                "text":
                                "Show me recent incident tickets "
                                "from Evolve Pharma"
                            }
                        ],

                        [
                            3,
                            {
                                "text":
                                "Summarize support case ID "
                                "981492 in markdown"
                            }
                        ],

                        [
                            4,
                            {
                                "text":
                                "Who are the support personnel "
                                "responsible for Evolve Pharma?"
                            }
                        ]
                    ],

                    inputs=[
                        example_num,
                        input
                    ],

                    elem_id="examples_table",
                    label=""
                )


        # ----------------------------------------------------
        # Right column
        # ----------------------------------------------------
        with gr.Column(
            scale=15,
            elem_id="col"
        ):

            chatbot = gr.Chatbot(
                value=startup_history,
                avatar_images=[
                    "assets/person.png",
                    "assets/chatbot.png"
                ],
                elem_id="chatbot",
                show_label=False
            )

            input.render()


    # ========================================================
    # Events
    # ========================================================

    user_msg = input.submit(
        display_user_message,
        [input, chatbot],
        [
            request_id,
            request_text,
            chatbot
        ]
    )


    input.submit(
        lambda x: gr.update(
            value={"text": ""}
        ),
        None,
        [input],
        queue=False
    )


    tool_msg = user_msg.then(
        display_thinking,
        chatbot,
        chatbot
    )


    ayb_msg = tool_msg.then(
        respond,
        [
            request_id,
            request_text,
            chatbot
        ],
        chatbot
    )


# ============================================================
# Launch
# ============================================================
print("Starting Gradio application")

demo.launch(
    server_port=int(
        os.getenv("CDSW_APP_PORT")
    ),

    server_name="127.0.0.1",

    debug=True,

    allowed_paths=[
        "/home/cdsw/assets/background.png"
    ]
)
