# Streamlit App with a Hugging Face Model - Chatbot

## Part I. Connect to an open source LLM (Hugging Face)

### 1. Choose the suitable model
Because of the relationship between the performance and volume of large language models, we need to choose a model of appropriate size. Here I selected the TinyLlama model after screening. Compared with some popular language models, the TinyLlama model has fewer parameters, but its performance is similar.
URL: https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0
According to the introduction on the web page, we directly choose `Use a pipeline as a high-level helper`
```
# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
```

### 2. Model Implementation
Here we define a function to get the response from the model:
```
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
```

## Part II. Create a Website Using Streamlit
### 3. Install necessary libraries and configure virtual environment
Next we need to configure `venv`. First, run `vanv\Scripts\activate` to start the virtual environment.
Then, install necessary libraries:
```
pip3 install streamlit
pip3 install torch
pip3 install transformers
```
Also, we can run `requirements.txt` instead.

### 4. Build the Website
Combined with the web design of general AI dialogue assistants, we first add a title and input field, then add a configuration bar on the left, and leave the remaining space for conversation records.
```
# Create column
col1, col2 = st.columns([1, 4])

# Place image in first column
with col1:
    st.image("C:/Users/37493/ids721week9/bot.png", width=100)  # Adjust image path and width

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
```

Then, we can use `markdown` to custom the style.
```
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

user_avatar = "C:/Users/37493/ids721week9/user_icon.png" 
bot_avatar = "C:/Users/37493/ids721week9/bot_icon.png" 
```

Last, we need to add basic function for the conversation:
```
# user input
user_input = st.text_input("Please type your questions here: ", key="user_input")

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
```

### 5. Local Test
Run `streamlit run app.py` in the terminal, we will see the following content, and the system will automatically pop up the web page:
![screenshot of streamlit run]()
![screenshot of web]()
![screenshot of test]()
After the local test, our web now is ready for deployment.

## Part III. Deploy model via Streamlit(accessible via browser)
### 6. 
我们可以直接通过网页右上角的`Deploy`按钮来直接部署这个网页。但是需要注意的是，我们需要注册账号并且提前把代码上传到github中托管。
Follow the instruction, your app will be deployed eventually. Keep in mind this process could take a while to finish.
![screenshot of deploy]()

## Part IV. EC2 Hosting
### 7. Created an EC2 Instance

 因为instace使用的     	
Keep in mind that t2.micro instances are very limited. We may run out of RAM memory when setting environment.       	
 To fix it, I would recommend you create a swap file. Swap files act as additional RAM memory, but it's a bit slow as it runs from the hard drive. However they are useful when you want to allocate a memory peak, as it is your case. The following example shows how to create a 5GB swap file:   
  	```
    sudo fallocate -l 1G ~/swapfile
sudo dd if=/dev/zero of=~/swapfile bs=1024 count=1048576
sudo chmod 600 ~/swapfile
sudo mkswap ~/swapfile
sudo swapon ~/swapfile
```
 

