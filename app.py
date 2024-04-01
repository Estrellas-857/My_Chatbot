import streamlit as st
import torch
from transformers import pipeline

# Create a text generation pipeline, specify the model and parameters for generating text

pipe = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# Create column
col1, col2 = st.columns([1, 4])

# Place image in first column
with col1:
    st.image("bot.png", width=100)  # Adjust image path and width

# Place title in second column
with col2:
    st.title("My Little Chatbot")

st.sidebar.title("About")
st.sidebar.info(
    """
    This is a chatbot based on the TinyLlama model. You can influence the style and content of your answer by adjusting the parameters below.
    """
)
st.sidebar.title("Configuration")
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.01, help="Control the randomness of the output. Lower values make the output more deterministic, higher values make the output more diverse.")
top_k = st.sidebar.slider("Top K", min_value=1, max_value=100, value=50, help="Limit the generation at each step to only consider the K characters with the highest probability.")
top_p = st.sidebar.slider("Top P", min_value=0.0, max_value=1.0, value=0.95, step=0.01, help="Dynamically select the Top K value to accumulate to a specific probability P, increasing the diversity of the output.")

st.sidebar.markdown("--------")  # Add a dividing line
st.sidebar.markdown("Powered by George Wang")
# Custom style
st.markdown("""
    <style>
    .bot-label {
        color: orange;
        font-size: 18px;
    }
    .message {
        font-family: sans-serif;
        padding: 5px;
        border-radius: 5px;
        margin: 10px 0;
        word-wrap: break-word;
        overflow-wrap: break-word;
        font-size: 18px;
    }
    .user-message {
        text-align: left;
    }
    .bot-message {
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True)

user_avatar = "user_icon.png" 
bot_avatar = "bot_icon.png" 

# Make sure the history exists
if 'history' not in st.session_state:
    st.session_state.history = []

def get_model_response(input_text, temperature, top_k, top_p):
    # Format input text and prepare it for sending to the model
    messages = [{"role": "user", "content": input_text}]
    # Apply chat templates to format messages and generate responses
    prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=temperature, top_k=top_k, top_p=top_p)
    # Extract generated text
    reply = outputs[0]["generated_text"]

    # Try to keep only the content after the last </assistant> tag
    parts = reply.split("<|assistant|>")
    if len(parts) > 1:
        reply = parts[-1]  # take the last part
    else:
        reply = parts[0]  # Without the </assistant> tag, it is assumed that the entire reply is required
    # Remove leading and trailing spaces
    reply = reply.strip()

    return reply

# Show conversation history
def show_history():
    for role, text in st.session_state.history:
        # Create columns to place avatars and text
        col1, col2 = st.columns([1, 8])
        # Use different styles to mark User and Bot comments
        if role == "You":
            avatar = user_avatar
            with col1:
                st.image(avatar, width=50)
            with col2:
                st.markdown(f"**{role}**: <br>{text}", unsafe_allow_html=True)
        else:
            avatar = bot_avatar
            with col1:
                st.image(avatar, width=50)
            with col2:
                st.markdown(f"<span class='bot-label'>**Bot**:</span> <br>{text}<br><br>", unsafe_allow_html=True)

# Functions that process input and update history
def handle_input():
    user_input = st.session_state.user_input
    if user_input:  # Make sure the input is not empty
        # Update history
        st.session_state.history.append(("User", user_input))
        # Get and add bot responses
        bot_response = get_model_response(user_input)
        st.session_state.history.append(("Bot", bot_response))
        # Clear the input box to prepare for next input
        st.session_state.user_input = ""

device_type = "CUDA" if torch.cuda.is_available() else "CPU"
device_name = torch.cuda.get_device_name(torch.cuda.current_device()) if device_type == "CUDA" else "N/A"
st.write(f"Current Device: {device_type} - {device_name}")

# user input
user_input = st.text_input("Please type your questions here: ", key="user_input")

# print('begin')

# Handle reply if user input is non-empty
if user_input.strip():
    with st.spinner('Little Bot is trying to find the answer, please hold on...'):
        # Call the model to get the response
        chat_response = get_model_response(user_input, temperature, top_k, top_p)
    # Update session history
    st.session_state.history.append(("You", user_input))
    st.session_state.history.append(("Bot", chat_response))
    st.success('Little Bot got the answer. Please scroll down to the bottom. ^_^')
    show_history()

# print('end')

