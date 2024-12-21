import streamlit as st
import os
from dotenv import load_dotenv  # 导入 dotenv 库
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage

# --- 加载 .env 文件中的变量 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_url = os.getenv("OPENAI_API_URL")


# # --- 初始化 ChatOpenAI ---
# llm = ChatOpenAI(model="gpt-4o", base_url=api_url, api_key=api_key, temperature=0)

# # --- 页面配置 ---
# st.set_page_config(page_title="chatbot", layout="wide")
# st.title("🤖 GPT-4o Chat Application")

# # --- 聊天记录 ---
# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# # --- 主聊天界面 ---
# st.header("Chat with GPT-4o")
# # 显示历史消息
# for message in st.session_state["messages"]:
#     if isinstance(message, HumanMessage):
#         with st.chat_message("user"):
#             st.markdown(message.content)
#     elif isinstance(message, AIMessage):
#         with st.chat_message("assistant"):
#             st.markdown(message.content)

# # 用户输入新消息
# prompt = st.chat_input("Type your message here...")
# if prompt:
#     # 用户输入
#     st.session_state["messages"].append(HumanMessage(content=prompt))
#     with st.chat_message("user"):
#         st.markdown(prompt)
    
#     # 发送到 GPT-4o 模型并获取响应
#     with st.spinner("GPT-4o is thinking..."):
#         ai_message = llm([HumanMessage(content=prompt)])
#     st.session_state["messages"].append(ai_message)
    
#     # 显示 GPT-4o 回复
#     with st.chat_message("assistant"):
#         st.markdown(ai_message.content)




# --- Sidebar for Navigation ---
# st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select a Page", ["GPT-4o", "GPT-3.5"])

# # --- Fixed API Configurations ---
# API_URL = "https://api.zetatechs.com/v1"

# # --- Page: GPT-4o ---
# if page == "GPT-4o":
#     st.title("🤖 GPT-4o Chat")
#     chat = ChatOpenAI(api_key=api_key, base_url=API_URL, model="gpt-4o", temperature=0)

#     # Chat Interface
#     if "messages" not in st.session_state:
#         st.session_state.messages = []

#     for msg in st.session_state.messages:
#         st.chat_message(msg["role"]).write(msg["content"])

#     prompt = st.chat_input("Ask GPT-4o...")
#     if prompt:
#         st.chat_message("user").write(prompt)
#         response = chat.invoke([HumanMessage(content=prompt)])
#         st.chat_message("assistant").write(response.content)
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         st.session_state.messages.append({"role": "assistant", "content": response.content})

# # --- Page: GPT-3.5 ---
# elif page == "GPT-3.5":
#     st.title("🤖 GPT-3.5 Chat")
#     chat = ChatOpenAI(api_key=api_key, base_url=API_URL, model="gpt-3.5-turbo", temperature=0.5)

#     prompt = st.chat_input("Ask GPT-3.5...")
#     if prompt:
#         st.write("### User:", prompt)
#         response = chat.invoke([HumanMessage(content=prompt)])
#         st.write("### GPT-3.5:", response.content)




# # app.py (简化代码示例)
# import streamlit as st

# st.set_page_config(page_title="Multi-Page GPT App")

# # st.sidebar.title("Navigation")
# page = st.sidebar.radio("Select a Page", ["GPT-4o", "GPT-3.5"])

# if page == "GPT-4o":
#     st.switch_page("pages/2_⚙️_OpenAI_GPT_4.py")
# elif page == "GPT-3.5":
#     st.switch_page("pages/3_⚙️_GPT_35.py")


import streamlit as st

# 设置页面配置
st.set_page_config(page_title="", layout="wide")

# 主内容
st.title("🤖 Welcome to FinChat Assistant")






with st.sidebar:
    #  st.divider()
     st.video("https://www.youtube.com/watch?v=4j2emMn7UaI")


