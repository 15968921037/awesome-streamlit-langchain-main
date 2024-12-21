############################################
import streamlit as st
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import AIMessage, HumanMessage


import os
from dotenv import load_dotenv  # 导入 dotenv 库
from langchain_openai import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from langchain_core.tools import tool
from sympy import sympify, symbols, Eq, solve
from scipy.stats import norm
import functools
import operator

from langchain_core.messages import BaseMessage, HumanMessage  # 导入消息相关的类
from langchain_openai.chat_models import ChatOpenAI  # 导入 OpenAI 的聊天模型类
from langgraph.prebuilt import create_react_agent  # 导入创建 React 风格代理的函数
from langgraph.graph import END, StateGraph, START  # 用于图形状态管理
from langchain_core.messages import HumanMessage  # 用于创建用户消息

#Define State 定义状态
import operator
from typing import Annotated, Sequence, TypedDict
from langchain_openai import ChatOpenAI
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults

from typing import List, Optional  # 用于类型注解，List表示列表，Optional表示可选类型
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser  # 用于解析 OpenAI 函数调用的输出
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 用于创建聊天提示模板和占位符
from langchain_openai import ChatOpenAI  # 用于与 OpenAI 的聊天接口交互


# --- 加载 .env 文件中的变量 ---
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
api_url = os.getenv("OPENAI_API_URL")

st.title("📻 Financial Current News Analysis")

llm = ChatOpenAI(model="gpt-4o", base_url=api_url, api_key=api_key, temperature=0)


# --- 定义 scrape_webpages 工具 ---
@tool
def scrape_webpages(urls: List[str]) -> str:
    """
    使用 WebBaseLoader 抓取指定网页内容并返回。
    参数:
    - urls: 一个包含网页 URL 的字符串列表。

    返回:
    - 一个包含所有抓取到网页内容的字符串，每个网页内容之间用两个换行符分隔。
    """
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [f'\n{doc.page_content}\n' for doc in docs]  # 拼接文档内容
    )



# 导入所需模块
from typing import List, Optional  # 用于类型注解，List表示列表，Optional表示可选类型
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser  # 用于解析 OpenAI 函数调用的输出
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 用于创建聊天提示模板和占位符
from langchain_openai import ChatOpenAI  # 用于与 OpenAI 的聊天接口交互

# 导入图形处理相关的模块
from langgraph.graph import END, StateGraph, START  # 用于图形状态管理
from langchain_core.messages import HumanMessage  # 用于创建用户消息

# 导入所需模块
from typing import List, Optional  # 用于类型注解，List表示列表，Optional表示可选类型
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser  # 用于解析 OpenAI 函数调用的输出
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 用于创建聊天提示模板和占位符
from langchain_openai import ChatOpenAI  # 用于与 OpenAI 的聊天接口交互

# 导入图形处理相关的模块
from langgraph.graph import END, StateGraph, START  # 用于图形状态管理
from langchain_core.messages import HumanMessage  # 用于创建用户消息

# 定义一个团队状态类，用于跟踪团队成员和任务状态
class CalcuTeamState(TypedDict):
    """
    ResearchTeamState 用于表示团队成员状态的字典结构。每个团队成员的消息会被记录下来，
    并且通过“next”字段来决定下一个应该执行任务的成员。
    """
    # 每个团队成员完成任务后会生成一条消息
    messages: Annotated[List[BaseMessage], operator.add]  # 消息列表，记录团队成员的所有输出
    # 团队成员的技能信息，以便每个成员了解其他成员的能力
    team_members: List[str]  # 团队成员列表
    # 用来路由任务，监督者会根据当前状态决定哪个成员继续执行任务
    next: str  # 当前轮到的成员，或者“FINISH”表示任务完成

# 定义代理节点的函数，用于调用具体的代理并返回结果
def agent_node(state, agent, name):
    result = agent.invoke(state)  # 调用代理，传入状态
    return {"messages": [HumanMessage(content=result["messages"][-1].content, name=name)]}  # 返回代理结果

# 创建一个团队监督者函数，作为基于LLM的路由器
def create_team_supervisor(llm: ChatOpenAI, system_prompt, members) -> str:
    """
    创建一个基于 LLM 的团队路由器，负责选择下一个需要执行任务的角色。
    参数:
    - llm: 使用的聊天模型，通常为 ChatOpenAI 类型
    - system_prompt: 系统提示，用于指导模型的行为
    - members: 任务团队的成员列表
    
    返回:
    - 返回包含路由逻辑和输出解析的链式对象
    """
    # 定义可选的角色（包括完成任务的选项）
    options = ["FINISH"] + members
    
    # 定义路由函数的结构
    function_def = {
        "name": "route",  # 函数名称
        "description": "Select the next role.",  # 函数描述
        "parameters": {
            "title": "routeSchema",  # 参数标题
            "type": "object",  # 参数类型
            "properties": {
                "next": {
                    "title": "Next",  # 路由到下一个角色的参数
                    "anyOf": [
                        {"enum": options},  # 可选择的角色选项，包括“FINISH”和所有团队成员
                    ],
                },
            },
            "required": ["next"],  # 必填项：next
        },
    }
    
    # 创建聊天提示模板，包含系统提示和团队成员信息
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),  # 系统消息，传入给模型的指令
            
            MessagesPlaceholder(variable_name="messages"),  # 占位符，用于插入实际的消息内容
            (
                "system",
                "Given the conversation above, who should act next?"
                " Or should we FINISH? Select one of: {options}",  # 提示模型选择下一个执行任务的角色
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(members))  # 将角色选项和成员信息传递到模板中
    
    # 返回一个组合了提示模板、函数绑定和输出解析的完整管道
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")  # 绑定函数并指定调用函数
        | JsonOutputFunctionsParser()  # 输出解析器，解析返回的 JSON 格式数据
    )


# 创建一个 TavilySearchResults 工具实例，用于执行搜索操作，设置最大返回结果为 5
tavily_tool = TavilySearchResults(max_results=5)

# 创建用于搜索的代理
search_agent = create_react_agent(llm, tools=[tavily_tool])  # 使用 TavilySearchResults 工具
search_node = functools.partial(agent_node, agent=search_agent, name="Search")

# 创建用于网页抓取的代理
research_agent = create_react_agent(llm, tools=[scrape_webpages])  # 使用 scrape_webpages 工具
research_node = functools.partial(agent_node, agent=research_agent, name="WebScraper")

supervisor_agent = create_team_supervisor(
    llm,
    "You're an expert in financial news analysis and interpretation, specializing in simplifying complex topics for beginners."
    "Your role is to manage conversations between staff members: WebScraper and Search."
    "Always prioritize WebScraper for gathering the latest financial news."
    "If WebScraper cannot provide an answer, delegate the task to Search. When finished, reply 'FINISH'."
    "please answer easy to understand for beginners.  Follow these guidelines:"
    "1.  Summarize the news topic in one or two sentences."
    "2.  Explain the key points step-by-step (use numbered points, e.g., 1, 2, 3)."
    "3.  Highlight the potential impact of the news in a simple way (e.g., This might affect prices, jobs, or investments)."
    "Your goal is to help users understand financial news in a straightforward and beginner-friendly manner.",
    ["WebScraper", "Search"],
)

# 创建团队状态类
class ResearchTeamState(TypedDict):
    messages: List[BaseMessage]
    team_members: List[str]
    next: str

# 定义团队图
task_graph = StateGraph(ResearchTeamState)
task_graph.add_node("WebScraper", research_node)
task_graph.add_node("Search", search_node)
task_graph.add_node("supervisor", supervisor_agent)

task_graph.add_edge("WebScraper", "supervisor")
task_graph.add_edge("Search", "supervisor")

task_graph.add_conditional_edges(
    "supervisor",
    lambda x: x["next"],
    {"WebScraper": "WebScraper", "Search": "Search", "FINISH": END},
)
task_graph.add_edge(START, "supervisor")
chain = task_graph.compile()

messages_key = "news_messages"  # 唯一键，确保各页面独立

# 初始化当前页面的消息记录
if messages_key not in st.session_state:
    st.session_state[messages_key] = []  # 初始化消

# --- 主聊天界面 ---
# st.header("Chat with GPT-4o")
# 显示历史消息
for message in st.session_state[messages_key]:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# 获取用户输入
prompt = st.chat_input("Ask your question...")
if prompt:
    # 显示用户输入
    st.session_state[messages_key].append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 LangGraph 并显示结果
    with st.spinner("Finchat is thinking..."):
        initial_state = {"messages": st.session_state[messages_key]}
        response = chain.invoke(initial_state)

    # 显示结果
    with st.chat_message("assistant"):
        st.markdown(response["messages"][-1].content)
        st.session_state[messages_key].append(AIMessage(content=response["messages"][-1].content))


      
#What is the step-by-step solution for calculating compound interest over 5 years at 5%?

# 清空聊天记录按钮
if st.button("Clear Chat", key=f"clear_{messages_key}"):
    st.session_state[messages_key].clear()



with st.sidebar:
     st.image("./images/your_image.png")
