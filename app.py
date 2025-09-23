import gradio as gr
import os
import json
from inference.react_agent import MultiTurnReactAgent

# --- Agent Setup ---
# This setup is based on the logic in `run_multi_react.py`

# Define the configuration for the language model
llm_cfg = {
    'model': "huggingface.co/gabriellarson/Tongyi-DeepResearch-30B-A3B-GGUF",
    'generate_cfg': {
        'max_input_tokens': 320000,
        'max_retries': 10,
        'temperature': 0.85,
        'top_p': 0.95,
        'presence_penalty': 1.1
    },
    'model_type': 'qwen_dashscope' # This might need adjustment for pure ollama
}

# Instantiate the agent
# The function_list enables the tools the agent can use.
# We are using the tools that were rewritten to be local.
agent = MultiTurnReactAgent(
    llm=llm_cfg,
    function_list=["search", "visit"]
)

# --- Gradio Interface Logic ---

def run_agent_and_stream_output(question: str):
    """
    Runs the agent with the given question and streams the output.
    This function is designed to be used with a Gradio interface.
    """
    if not question:
        yield "Please enter a question."
        return

    # The agent's _run method expects a specific data structure
    task_data = {
        'item': {'question': question, 'answer': ''},
        'planning_port': 11434, # ollama default port
    }

    # The _run method returns the final result, but we want to stream thoughts.
    # The thoughts are printed within the agent's methods. To capture them for the UI,
    # we would need to refactor the agent to yield them.
    # For this first version, we will run the agent and capture the final result.
    # Streaming intermediate thoughts will be a future improvement.

    yield "Agent is thinking... (Streaming of intermediate thoughts is not yet implemented)"

    try:
        # The _run method in the provided code is not a generator.
        # It directly returns the final result dictionary.
        # We will call it and format the output.
        result = agent._run(task_data, model=llm_cfg['model'])

        # The result is a dictionary containing the full conversation history.
        # Let's format it for display.
        formatted_output = "--- Agent Run Complete ---\n\n"

        final_answer = result.get("prediction", "No final answer found.")
        formatted_output += f"**Final Answer:**\n{final_answer}\n\n"

        formatted_output += "**Full Conversation History:**\n"
        for message in result.get("messages", []):
            role = message.get("role", "unknown")
            content = message.get("content", "").replace('<', '&lt;').replace('>', '&gt;')
            formatted_output += f"**{role.capitalize()}:**\n```\n{content}\n```\n\n"

        yield formatted_output

    except Exception as e:
        yield f"An error occurred: {e}"


# --- Gradio App Definition ---

with gr.Blocks() as demo:
    gr.Markdown("# DeepResearch Agent UI")
    gr.Markdown("Enter your question or task for the agent below.")

    with gr.Row():
        question_input = gr.Textbox(label="Your Question", placeholder="e.g., Introduce Alibaba web agent")

    run_button = gr.Button("Run Agent")

    with gr.Row():
        output_display = gr.Markdown(label="Agent Output")

    run_button.click(
        fn=run_agent_and_stream_output,
        inputs=question_input,
        outputs=output_display
    )

if __name__ == "__main__":
    demo.launch()
