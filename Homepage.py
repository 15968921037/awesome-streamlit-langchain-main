# # cd /Users/alin/Downloads/awesome-streamlit-langchain-main
# # git init
# # git add .
# # git status
# # git commit -m "Initial commit"
# # git remote add origin https://github.com/15968921037/awesome-streamlit-langchain-main.git
# # git branch -M main # 如果是第一次推送，确保分支名为 main
# # git push -u origin main
# ##export PATH="/Users/alin/Downloads/awesome-streamlit-langchain-main/venv/bin:$PATH"
# ##streamlit run Homepage.py  

# import streamlit as st
# # from langchain.chat_models import ChatOpenAI
# from openai import OpenAI
# from langchain_openai import ChatOpenAI
# from langchain.schema import (
#     AIMessage,
#     HumanMessage,
#     SystemMessage
# )

# # Initialize the ChatOpenAI object
# chat = None

# if "OPENAI_API_KEY" not in st.session_state:
#     st.session_state["OPENAI_API_KEY"] = ""
# elif st.session_state["OPENAI_API_KEY"] != "":
#     chat = ChatOpenAI(openai_api_key=st.session_state["OPENAI_API_KEY"])

# if "PINECONE_API_KEY" not in st.session_state:
#     st.session_state["PINECONE_API_KEY"] = ""

# if "PINECONE_ENVIRONMENT" not in st.session_state:
#     st.session_state["PINECONE_ENVIRONMENT"] = ""

# st.set_page_config(page_title="Welcome to ASL", layout="wide")

# st.title("🤠 Welcome to ASL")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if chat:
#     with st.container():
#         st.header("Chat with GPT")
#         for message in st.session_state["messages"]:
#             if isinstance(message, HumanMessage):
#                 with st.chat_message("user"):
#                     st.markdown(message.content)
#             elif isinstance(message, AIMessage):
#                 with st.chat_message("assistant"):
#                     st.markdown(message.content)
#         prompt = st.chat_input("Type something...")
#         if prompt:
#             st.session_state["messages"].append(HumanMessage(content=prompt))
#             with st.chat_message("user"):
#                 st.markdown(prompt)
#             ai_message = chat([HumanMessage(content=prompt)])
#             st.session_state["messages"].append(ai_message)
#             with st.chat_message("assistant"):
#                 st.markdown(ai_message.content)
# else:
#     with st.container():
#         st.warning("Please set your OpenAI API key in the settings page.")




import streamlit as st
import os
from dotenv import load_dotenv  # 导入 dotenv 库
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

# --- 加载 .env 文件中的变量 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_url = os.getenv("OPENAI_API_URL")


# --- 初始化 ChatOpenAI ---
llm = ChatOpenAI(model="gpt-4o", base_url=api_url, api_key=api_key, temperature=0)

# --- 页面配置 ---
st.set_page_config(page_title="GPT-4o Chat App", layout="wide")
st.title("🤖 GPT-4o Chat Application")

# --- 聊天记录 ---
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# --- 主聊天界面 ---
st.header("Chat with GPT-4o")
# 显示历史消息
for message in st.session_state["messages"]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 用户输入新消息
prompt = st.chat_input("Type your message here...")
if prompt:
    # 用户输入
    st.session_state["messages"].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # 发送到 GPT-4o 模型并获取响应
    with st.spinner("GPT-4o is thinking..."):
        ai_message = llm([HumanMessage(content=prompt)])
    st.session_state["messages"].append(ai_message)
    
    # 显示 GPT-4o 回复
    with st.chat_message("assistant"):
        st.markdown(ai_message.content)
