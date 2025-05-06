import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
import nest_asyncio
from streamlit.components.v1 import html
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "app.log")
nest_asyncio.apply()

def validate_input(user_query):
    if not user_query:
        return False, "輸入不能為空"
    if len(user_query) < 3:
        return False, "輸入過短，至少3個字"
    if len(user_query) > 200:
        return False, "輸入過長，最多200個字"
    return True, ""

def render_copy_button(content, key):
    escaped_content = content.replace("`", "\\`").replace("\n", "\\n")
    html_code = f"""
    <button onclick="navigator.clipboard.writeText(`{escaped_content}`)"
            title="複製回應"
            style="border: none; background: none; cursor: pointer; font-size: 20px;">
        📋
    </button>
    """
    html(html_code, height=30)

def render_new_conversation_button():
    if st.button("🆕 新對話", help="開始新對話"):
        st.session_state.chat_history = []
        st.session_state.conversation_context = []
        st.session_state.context_timestamps = []
        st.session_state.thread_cache = {}
        st.session_state.last_user_query = None
        st.session_state.awaiting_response = False
        logger.info("New conversation started, session state cleared")
        st.rerun()

async def main():
    st.set_page_config(page_title="Social Media Chat Bot", layout="centered")
    st.title("Social Media Chat Bot")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "thread_cache" not in st.session_state:
        st.session_state.thread_cache = {}
    if "awaiting_response" not in st.session_state:
        st.session_state.awaiting_response = False
    if "conversation_context" not in st.session_state:
        st.session_state.conversation_context = []
    if "context_timestamps" not in st.session_state:
        st.session_state.context_timestamps = []
    if "last_selected_source" not in st.session_state:
        st.session_state.last_selected_source = None
    if "page_reload_logged" not in st.session_state:
        st.session_state.page_reload_logged = True
        logger.info(f"Page reloaded, last_selected_source: {st.session_state.get('last_selected_source', 'None')}")

    source_map = {
        "LIHKG - 吹水台": {"source": "lihkg", "cat_id": "1"},
        "LIHKG - 熱門台": {"source": "lihkg", "cat_id": "2"},
        "LIHKG - 時事台": {"source": "lihkg", "cat_id": "5"},
        "LIHKG - 上班台": {"source": "lihkg", "cat_id": "14"},
        "LIHKG - 財經台": {"source": "lihkg", "cat_id": "15"},
        "LIHKG - 成人台": {"source": "lihkg", "cat_id": "29"},
        "LIHKG - 創意台": {"source": "lihkg", "cat_id": "31"},
        "Reddit - wallstreetbets": {"source": "reddit", "subreddit": "wallstreetbets"},
        "Reddit - personalfinance": {"source": "reddit", "subreddit": "personalfinance"},
        "Reddit - investing": {"source": "reddit", "subreddit": "investing"},
        "Reddit - stocks": {"source": "reddit", "subreddit": "stocks"},
        "Reddit - options": {"source": "reddit", "subreddit": "options"}
    }

    def on_source_change():
        selected_source = st.session_state.get("source_select", "未知來源")
        logger.info(f"Source selectbox changed to {selected_source}")

    col1, col2 = st.columns([3, 1])
    with col1:
        try:
            selected_source = st.selectbox(
                "選擇數據來源",
                options=list(source_map.keys()),
                index=0,
                key="source_select",
                on_change=on_source_change
            )
            source_info = source_map.get(selected_source, {"source": "lihkg", "cat_id": "1"})
            source_type = source_info.get("source", "lihkg")
            if source_type == "lihkg":
                source_id = source_info.get("cat_id", "1")
                selected_cat = selected_source
            else:
                source_id = source_info.get("subreddit", "stocks")
                selected_cat = selected_source
        except Exception as e:
            logger.error(f"數據來源選擇錯誤：{str(e)}")
            selected_cat = "LIHKG - 吹水台"
            source_type = "lihkg"
            source_id = "1"
    with col2:
        render_new_conversation_button()

    if st.session_state.last_selected_source != selected_source:
        if st.button("確認切換數據來源並清除歷史"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.context_timestamps = []
            st.session_state.thread_cache = {}
            st.session_state.last_user_query = None
            logger.info(f"Source changed to {selected_source}, cleared conversation history")
        st.session_state.last_selected_source = selected_source
    else:
        logger.info(f"Source unchanged: {selected_source}, preserving conversation history")

    st.write(f"當前數據來源：{selected_cat}")

    for idx, chat in enumerate(st.session_state.chat_history):
        with st.chat_message("user"):
            st.markdown(chat["question"])
        with st.chat_message("assistant"):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                st.markdown(chat["answer"])
            with col2:
                render_copy_button(chat["answer"], key=f"copy_{idx}")

    user_query = st.chat_input("請輸入 LIHKG 或 Reddit 話題或一般問題")
    if user_query and not st.session_state.awaiting_response:
        is_valid, error_message = validate_input(user_query)
        if not is_valid:
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history.append({"question": user_query, "answer": error_message})
            return

        logger.info(f"User query: {user_query}, source: {selected_source}, source_type: {source_type}, source_id: {source_id}")
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.awaiting_response = True

        current_time = time.time()
        valid_context = []
        valid_timestamps = []
        for msg, ts in zip(st.session_state.conversation_context, st.session_state.context_timestamps):
            if current_time - ts < 7200:
                valid_context.append(msg)
                valid_timestamps.append(ts)
        st.session_state.conversation_context = valid_context[-20:]
        st.session_state.context_timestamps = valid_timestamps[-20:]
        st.session_state.chat_history = st.session_state.chat_history[-20:]

        status_text = st.empty()
        progress_bar = st.progress(0)

        def update_progress(message, progress):
            status_text.write(f"正在處理... {message}")
            progress_bar.progress(min(max(progress, 0.0), 1.0))

        try:
            update_progress("正在初始化", 0.0)

            st.session_state.chat_history.append({"question": user_query, "answer": ""})
            st.session_state.last_user_query = user_query

            update_progress("正在分析問題意圖", 0.1)
            analysis = await analyze_and_screen(
                user_query=user_query,
                source_name=selected_cat,
                source_id=source_id,
                source_type=source_type,
                conversation_context=st.session_state.conversation_context
            )
            logger.info(f"Analysis completed: intent={analysis.get('intent')}")

            # 檢查緩存，避免重複抓取
            cache_key = f"{source_id}_topics"
            if cache_key in st.session_state.thread_cache:
                cached_data = st.session_state.thread_cache[cache_key]
                if time.time() - cached_data["timestamp"] < 300:  # 緩存 5 分鐘
                    logger.info(f"使用 app 層緩存數據，來源：{source_id}")
                    result = cached_data["data"]
                else:
                    update_progress("正在處理查詢", 0.2)
                    result = await process_user_question(
                        user_query=user_query,
                        selected_source=selected_cat,
                        source_id=source_id,
                        source_type=source_type,
                        analysis=analysis,
                        conversation_context=st.session_state.conversation_context,
                        progress_callback=update_progress
                    )
                    st.session_state.thread_cache[cache_key] = {
                        "timestamp": time.time(),
                        "data": result
                    }
            else:
                update_progress("正在處理查詢", 0.2)
                result = await process_user_question(
                    user_query=user_query,
                    selected_source=selected_cat,
                    source_id=source_id,
                    source_type=source_type,
                    analysis=analysis,
                    conversation_context=st.session_state.conversation_context,
                    progress_callback=update_progress
                )
                st.session_state.thread_cache[cache_key] = {
                    "timestamp": time.time(),
                    "data": result
                }

            response = ""
            with st.chat_message("assistant"):
                col1, col2 = st.columns([0.95, 0.05])
                with col1:
                    grok_container = st.empty()
                with col2:
                    copy_container = st.empty()
                update_progress("正在生成回應", 0.9)
                logger.info(f"Starting stream_grok3_response for query: {user_query}, intent: {analysis.get('intent')}")
                async for chunk in stream_grok3_response(
                    user_query=user_query,
                    metadata=[{"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"), "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0} for item in result.get("thread_data", [])],
                    thread_data={item["thread_id"]: item for item in result.get("thread_data", [])},
                    processing=analysis.get("processing", "general"),
                    selected_source=selected_cat,
                    conversation_context=st.session_state.conversation_context,
                    needs_advanced_analysis=analysis.get("needs_advanced_analysis", False),
                    reason=analysis.get("reason", ""),
                    filters=analysis.get("filters", {}),
                    source_id=source_id,
                    source_type=source_type
                ):
                    response += chunk
                    grok_container.markdown(response)
                if not response:
                    logger.warning(f"No response generated for query: {user_query}")
                    response = "無法生成回應，請稍後重試。"
                    grok_container.markdown(response)
                copy_container.empty()
                render_copy_button(response, key=f"copy_new_{len(st.session_state.chat_history)}")

            st.session_state.chat_history[-1]["answer"] = response
            st.session_state.conversation_context.append({"role": "user", "content": user_query})
            st.session_state.conversation_context.append({"role": "assistant", "content": response})
            st.session_state.context_timestamps.append(time.time())
            st.session_state.context_timestamps.append(time.time())
            st.session_state.conversation_context = st.session_state.conversation_context[-20:]
            st.session_state.context_timestamps = st.session_state.context_timestamps[-20:]
            update_progress("完成", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()

        except Exception as e:
            error_message = f"處理失敗：{str(e)}"
            logger.error(f"Error processing query: {user_query}, error: {str(e)}")
            with st.chat_message("assistant"):
                st.markdown(error_message)
            st.session_state.chat_history[-1]["answer"] = error_message
            update_progress("處理失敗", 1.0)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
        finally:
            st.session_state.awaiting_response = False

if __name__ == "__main__":
    asyncio.run(main())
