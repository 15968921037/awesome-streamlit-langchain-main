
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
)

# --- 固定的 API Key 和 URL ---
api_key = "sk-HejZVE2sOL2vgP7Ft6Qoekwdr4vuTzRXfpKwIObAi2sonITy"
api_url = "https://api.zetatechs.com/v1"

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
