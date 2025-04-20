"""
Streamlit 聊天介面模組，提供 LIHKG 數據查詢和顯示功能。
負責用戶交互、聊天記錄管理和速率限制狀態顯示。
主要函數：
- main：初始化應用，處理用戶輸入，渲染介面。
"""

import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
import nest_asyncio
import logging
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question
from lihkg_api import get_category_name

# 應用 asyncio 補丁，允許在 Streamlit 中運行異步函數
nest_asyncio.apply()

# 配置日誌記錄器，用於追蹤用戶交互和錯誤
logger = logging.getLogger(__name__)

# 香港時區，用於顯示時間
HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

async def main():
    """
    主函數，初始化 Streamlit 應用，處理用戶輸入並渲染聊天介面。
    管理聊天記錄、緩存、速率限制狀態。
    """
    # 設置頁面標題
    st.title("LIHKG 聊天介面")

    # 初始化 session_state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_cache" not in st.session_state:
        st.session_state.thread_cache = {}
    if "rate_limit_until" not in st.session_state:
        st.session_state.rate_limit_until = 0
    if "request_counter" not in st.session_state:
        st.session_state.request_counter = 0
    if "last_reset" not in st.session_state:
        st.session_state.last_reset = time.time()
    if "last_user_query" not in st.session_state:
        st.session_state.last_user_query = None
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    
    # 分類選擇
    cat_id_map = {
        "吹水台": 1, "熱門台": 2, "時事台": 5, "上班台": 14,
        "財經台": 15, "成人台": 29, "創意台": 31
    }
    selected_cat = st.selectbox("選擇分類", options=list(cat_id_map.keys()), index=0)
    cat_id = cat_id_map[selected_cat]
    
    # 顯示速率限制狀態
    st.markdown("#### 速率限制狀態")
    st.markdown(f"- 請求計數: {st.session_state.request_counter}")
    st.markdown(f"- 最後重置: {datetime.fromtimestamp(st.session_state.last_reset, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
    st.markdown(f"- 速率限制解除: {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S if st.session_state.rate_limit_until > time.time() else '無限制'}")
    
    # 顯示聊天記錄
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            st.markdown(chat["answer"])
    
    # 用戶輸入
    user_question = st.chat_input("請輸入 LIHKG 話題（例如：有哪些搞笑話題？）")
    if user_question and not st.session_state.awaiting_response:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.awaiting_response = True
        
        with st.spinner("正在處理..."):
            try:
                # 檢查速率限制
                if time.time() < st.session_state.rate_limit_until:
                    error_message = f"速率限制中，請在 {datetime.fromtimestamp(st.session_state.rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S} 後重試。"
                    with st.chat_message("assistant"):
                        st.markdown(error_message)
                    st.session_state.chat_history.append({"question": user_question, "answer": error_message})
                    st.session_state.awaiting_response = False
                    return
                
                # 重置聊天記錄（若問題變化較大）
                if not st.session_state.last_user_query or len(set(user_question.split()).intersection(set(st.session_state.last_user_query.split()))) < 2:
                    st.session_state.chat_history = [{"question": user_question, "answer": ""}]
                    st.session_state.thread_cache = {}
                    st.session_state.last_user_query = user_question
                
                # 分析問題並篩選帖子
                analysis = await analyze_and_screen(user_query=user_question, cat_name=selected_cat, cat_id=cat_id)
                
                # 若無相關分類，直接生成回應
                if not analysis.get("category_ids"):
                    response = ""
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(user_question, [], {}, "summarize"):
                            response += chunk
                            grok_container.markdown(response)
                    st.session_state.chat_history[-1]["answer"] = response
                    st.session_state.awaiting_response = False
                    return
                
                # 處理用戶問題
                result = await process_user_question(
                    user_question=user_question, selected_cat=selected_cat, cat_id=cat_id, analysis=analysis,
                    request_counter=st.session_state.request_counter, last_reset=st.session_state.last_reset,
                    rate_limit_until=st.session_state.rate_limit_until
                )
                
                # 更新速率限制狀態
                st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                
                thread_data = result.get("thread_data", [])
                rate_limit_info = result.get("rate_limit_info", [])
                question_cat = result.get("selected_cat", selected_cat)
                
                # 若無帖子數據
                if not thread_data:
                    answer = f"在 {question_cat} 中未找到符合條件的帖子。"
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    st.session_state.chat_history[-1]["answer"] = answer
                    st.session_state.awaiting_response = False
                    return
                
                # 準備回應
                post_limit = analysis.get("post_limit", 2)
                thread_data = thread_data[:post_limit]
                theme = analysis.get("theme", "相關")
                response = f"以下分享{post_limit}個被認為『{theme}』的帖子：\n\n"
                metadata = [
                    {
                        "thread_id": item["thread_id"], "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"),
                        "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0)
                    } for item in thread_data
                ]
                for meta in metadata:
                    response += f"帖子 ID: {meta['thread_id']}\n標題: {meta['title']}\n"
                response += "\n"
                
                # 顯示回應
                with st.chat_message("assistant"):
                    grok_container = st.empty()
                    async for chunk in stream_grok3_response(user_question, metadata, {item["thread_id"]: item for item in thread_data}, analysis["processing"]):
                        response += chunk
                        grok_container.markdown(response)
                
                logger.info(f"Processed: category={question_cat}, threads={len(thread_data)}, rate_limit={rate_limit_info}")
                
                # 進階分析
                analysis_advanced = await analyze_and_screen(
                    user_query=user_question, cat_name=question_cat, cat_id=cat_id, thread_titles=None,
                    metadata=metadata, thread_data={item["thread_id"]: item for item in thread_data}, is_advanced=True
                )
                if analysis_advanced.get("needs_advanced_analysis"):
                    result = await process_user_question(
                        user_question=user_question, selected_cat=question_cat, cat_id=cat_id, analysis=analysis,
                        request_counter=st.session_state.request_counter, last_reset=st.session_state.last_reset,
                        rate_limit_until=st.session_state.rate_limit_until, is_advanced=True,
                        previous_thread_ids=[str(item["thread_id"]) for item in thread_data],
                        previous_thread_data={item["thread_id"]: item for item in thread_data}
                    )
                    st.session_state.request_counter = result.get("request_counter", st.session_state.request_counter)
                    st.session_state.last_reset = result.get("last_reset", st.session_state.last_reset)
                    st.session_state.rate_limit_until = result.get("rate_limit_until", st.session_state.rate_limit_until)
                    
                    thread_data_advanced = result.get("thread_data", [])
                    rate_limit_info = result.get("rate_limit_info", [])
                    
                    if thread_data_advanced:
                        for item in thread_data_advanced:
                            thread_id = str(item["thread_id"])
                            st.session_state.thread_cache[thread_id] = {"data": item, "timestamp": time.time()}
                        
                        metadata_advanced = [
                            {
                                "thread_id": item["thread_id"], "title": item["title"],
                                "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"),
                                "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0)
                            } for item in thread_data_advanced
                        ]
                        response += f"\n\n更深入的『{theme}』帖子分析：\n\n"
                        for meta in metadata_advanced:
                            response += f"帖子 ID: {meta['thread_id']}\n標題: {meta['title']}\n"
                        response += "\n"
                        async for chunk in stream_grok3_response(user_question, metadata_advanced, {item["thread_id"]: item for item in thread_data_advanced}, analysis["processing"]):
                            response += chunk
                            grok_container.markdown(response)
                
                st.session_state.chat_history[-1]["answer"] = response
                st.session_state.last_user_query = user_question
                st.session_state.awaiting_response = False
            
            except Exception as e:
                error_message = f"處理失敗：{str(e)}"
                logger.error(f"Processing error: {str(e)}")
                with st.chat_message("assistant"):
                    st.markdown(error_message)
                st.session_state.chat_history[-1]["answer"] = error_message
                st.session_state.awaiting_response = False
    
    # 處理後續指令
    if st.session_state.awaiting_response and st.session_state.chat_history[-1]["answer"]:
        response_input = st.chat_input("輸入指令（修改分類、ID 數字、結束）：")
        if response_input:
            response_input = response_input.strip().lower()
            if response_input == "結束":
                final_answer = "分析結束，感謝使用！"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": "結束", "answer": final_answer})
                st.session_state.awaiting_response = False
            elif response_input == "修改分類":
                final_answer = "請選擇新分類並輸入問題。"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": "修改分類", "answer": final_answer})
                st.session_state.awaiting_response = False
            elif response_input.isdigit():
                thread_id = response_input
                thread_data = st.session_state.chat_history[-1].get("thread_data", [])
                if thread_id in [str(item["thread_id"]) for item in thread_data]:
                    response = f"帖子 ID: {thread_id}\n\n"
                    with st.chat_message("assistant"):
                        grok_container = st.empty()
                        async for chunk in stream_grok3_response(
                            st.session_state.last_user_query,
                            [item for item in thread_data if str(item["thread_id"]) == thread_id],
                            {thread_id: next(item for item in thread_data if str(item["thread_id"]) == thread_id)},
                            "summarize"
                        ):
                            response += chunk
                            grok_container.markdown(response)
                    st.session_state.chat_history.append({"question": f"ID {thread_id}", "answer": response})
                else:
                    final_answer = f"無效帖子 ID {thread_id}。"
                    with st.chat_message("assistant"):
                        st.markdown(final_answer)
                    st.session_state.chat_history.append({"question": f"ID {thread_id}", "answer": final_answer})
                st.session_state.awaiting_response = False
            else:
                final_answer = "請輸入有效指令：修改分類、ID 數字、結束"
                with st.chat_message("assistant"):
                    st.markdown(final_answer)
                st.session_state.chat_history.append({"question": response_input, "answer": final_answer})

if __name__ == "__main__":
    asyncio.run(main())
