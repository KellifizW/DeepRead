import streamlit as st
import nest_asyncio
import asyncio
import time
from datetime import datetime
import pytz
from data_processor import DataProcessor
from lihkg_client import LIHKGClient
from grok3_client import Grok3Client

# 解决Streamlit的异步兼容问题
nest_asyncio.apply()

# 初始化全局配置
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
CATEGORIES = {
    "吹水台": 1, "熱門台": 2, "時事台": 5,
    "上班台": 14, "財經台": 15, "成人台": 29, "創意台": 31
}

def init_session_state():
    """初始化session state"""
    if "chat" not in st.session_state:
        st.session_state.chat = {
            "history": [],
            "last_query": None,
            "awaiting_response": False,
            "rate_limit": {
                "counter": 0,
                "last_reset": time.time(),
                "until": 0
            }
        }

async def main():
    # 初始化客户端
    lihkg_client = LIHKGClient()
    grok_client = Grok3Client()
    processor = DataProcessor(lihkg_client, grok_client)

    # 页面布局
    st.sidebar.title("LIHKG分析工具")
    page = st.sidebar.radio("選擇頁面", ["聊天介面", "測試頁面"])

    if page == "聊天介面":
        await chat_page(processor)
    elif page == "測試頁面":
        await test_page(lihkg_client)

async def chat_page(processor: DataProcessor):
    init_session_state()
    st.title("📊 LIHKG話題分析")

    # 分类选择
    selected_cat = st.selectbox(
        "選擇討論區分類",
        options=list(CATEGORIES.keys()),
        index=0
    )
    cat_id = CATEGORIES[selected_cat]

    # 显示速率限制状态
    with st.expander("速率限制狀態"):
        st.json(st.session_state.chat["rate_limit"])

    # 显示历史对话
    for msg in st.session_state.chat["history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # 处理用户输入
    user_input = st.chat_input("輸入想查詢的話題（例如：最近有什麼熱門討論？）")
    if user_input and not st.session_state.chat["awaiting_response"]:
        await process_user_input(
            processor=processor,
            query=user_input,
            cat_id=cat_id,
            cat_name=selected_cat
        )

async def process_user_input(processor: DataProcessor, query: str, cat_id: int, cat_name: str):
    """处理用户查询"""
    st.session_state.chat["awaiting_response"] = True
    st.session_state.chat["history"].append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # 1. 分析问题
        with st.spinner("正在分析問題..."):
            analysis = await processor.grok.analyze_question(query, cat_name, cat_id)

        # 2. 获取数据
        with st.spinner("抓取討論區數據..."):
            data = await processor.process_user_question(query, cat_id, analysis)

        # 3. 生成流式响应
        full_response = ""
        async for chunk in processor.grok.stream_response(
            query,
            metadata=data["metadata"],
            thread_data=data["content"],
            processing=analysis["processing"]
        ):
            full_response += chunk
            response_placeholder.markdown(full_response)

    # 更新对话状态
    st.session_state.chat["history"].append({"role": "assistant", "content": full_response})
    st.session_state.chat["last_query"] = query
    st.session_state.chat["awaiting_response"] = False
    st.session_state.chat["rate_limit"] = processor.lihkg.get_rate_limit_status()

async def test_page(lihkg_client: LIHKGClient):
    """测试页面实现"""
    st.title("LIHKG數據測試")
    # 实现内容与原始test_page.py一致
    # ...

if __name__ == "__main__":
    asyncio.run(main())