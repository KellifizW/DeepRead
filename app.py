import streamlit as st
import asyncio
import time
from datetime import datetime
import pytz
import nest_asyncio
from streamlit.components.v1 import html
from grok_processing import analyze_and_screen, stream_grok3_response, process_user_question, clean_cache
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
        logger.info("新對話開始，清除會話狀態")
        st.rerun()

def estimate_tokens(text):
    return len(text) // 4  # 粗略估計：1 token ≈ 4 字符

def get_source_info(selected_source, source_map):
    try:
        source_info = source_map.get(selected_source, {"source": "lihkg", "cat_id": "1"})
        source_type = source_info.get("source", "lihkg")
        source_id = source_info.get("cat_id" if source_type == "lihkg" else "subreddit", "1")
        selected_cat = selected_source
    except Exception as e:
        logger.error(f"數據來源選擇錯誤：{str(e)}")
        st.error(f"無法加載數據來源：{str(e)}，請重試")
        source_type, source_id, selected_cat = "lihkg", "1", "LIHKG - 吹水台"
    return source_type, source_id, selected_cat

async def fetch_thread_data(user_query, selected_cat, source_id, source_type, analysis, context):
    cache_key = f"{source_id}_{user_query[:50]}_{','.join([i['intent'] for i in analysis.get('intents', [])])}"
    if cache_key in st.session_state.thread_cache and time.time() - st.session_state.thread_cache[cache_key]["timestamp"] < 300:
        logger.info(f"使用緩存數據，來源：{source_id}，查詢：{user_query}")
        return st.session_state.thread_cache[cache_key]["data"], 0.35
    result = await process_user_question(
        user_query=user_query,
        selected_source=selected_cat,
        source_id=source_id,
        source_type=source_type,
        analysis=analysis,
        conversation_context=context,
        progress_callback=lambda msg, prog, details=None: update_progress(msg, 0.25 + prog * 0.30, source_type, details)
    )
    st.session_state.thread_cache[cache_key] = {"timestamp": time.time(), "data": result}
    return result, 0.55

def update_progress(message, progress, source_type=None, details=None):
    if source_type == "reddit" and details:
        if "current_comments" in details and "total_comments" in details:
            message += f" ({details['current_comments']}/{details['total_comments']} 條評論)"
        elif "wait_time" in details:
            message += f" (等待速率限制 {details['wait_time']:.2f} 秒)"
    elif source_type == "lihkg" and details:
        if "current_page" in details and "total_pages" in details:
            message += f" (第 {details['current_page']}/{details['total_pages']} 頁)"
        elif "current_thread" in details and "total_threads" in details:
            message += f" (第 {details['current_thread']}/{details['total_threads']} 帖子)"
        elif "wait_time" in details:
            message += f" (等待速率限制 {details['wait_time']:.2f} 秒)"
    status_text.write(f"正在處理：{message}")
    progress_bar.progress(min(max(progress, 0.0), 1.0))

async def main():
    st.set_page_config(page_title="社交媒體聊天機器人", layout="centered")
    st.title("社交媒體聊天機器人")

    # 初始化會話狀態
    for key, default in [
        ("chat_history", []), ("thread_cache", {}), ("awaiting_response", False),
        ("conversation_context", []), ("context_timestamps", []), ("last_selected_source", None)
    ]:
        if key not in st.session_state:
            st.session_state[key] = default
    if not st.session_state.get("page_reload_logged", False):
        st.session_state.page_reload_logged = True
        logger.info(f"頁面重新加載，上次來源：{st.session_state.get('last_selected_source', '無')}")

    source_map = {
        "LIHKG - 吹水台": {"source": "lihkg", "cat_id": "1"}, "LIHKG - 熱門台": {"source": "lihkg", "cat_id": "2"},
        "LIHKG - 時事台": {"source": "lihkg", "cat_id": "5"}, "LIHKG - 上班台": {"source": "lihkg", "cat_id": "14"},
        "LIHKG - 財經台": {"source": "lihkg", "cat_id": "15"}, "LIHKG - 成人台": {"source": "lihkg", "cat_id": "29"},
        "LIHKG - 創意台": {"source": "lihkg", "cat_id": "31"}, "Reddit - wallstreetbets": {"source": "reddit", "subreddit": "wallstreetbets"},
        "Reddit - personalfinance": {"source": "reddit", "subreddit": "personalfinance"}, "Reddit - investing": {"source": "reddit", "subreddit": "investing"},
        "Reddit - stocks": {"source": "reddit", "subreddit": "stocks"}, "Reddit - options": {"source": "reddit", "subreddit": "options"}
    }

    col1, col2 = st.columns([3, 1])
    with col1:
        selected_source = st.selectbox(
            "選擇數據來源", options=list(source_map.keys()), index=0, key="source_select",
            on_change=lambda: logger.info(f"來源變更為 {st.session_state.get('source_select', '未知來源')}")
        )
        source_type, source_id, selected_cat = get_source_info(selected_source, source_map)
    with col2:
        render_new_conversation_button()

    if st.session_state.last_selected_source != selected_source:
        if st.button("確認切換數據來源並清除歷史"):
            st.session_state.chat_history = []
            st.session_state.conversation_context = []
            st.session_state.context_timestamps = []
            st.session_state.thread_cache = {}
            st.session_state.last_user_query = None
            logger.info(f"來源切換至 {selected_source}，清除對話歷史")
        st.session_state.last_selected_source = selected_source
    else:
        logger.info(f"來源未變：{selected_source}")

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
    if not (user_query and not st.session_state.awaiting_response):
        return

    is_valid, error_message = validate_input(user_query)
    if not is_valid:
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.chat_history.append({"question": user_query, "answer": error_message})
        return

    logger.info(f"用戶查詢：{user_query}，來源：{selected_source}，類型：{source_type}，ID：{source_id}")
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.awaiting_response = True

    clean_cache(max_age=3600)
    current_time = time.time()
    token_limit, total_tokens = 100000, 0
    valid_context, valid_timestamps = [], []
    for msg, ts in zip(st.session_state.conversation_context, st.session_state.context_timestamps):
        if current_time - ts < 7200:
            msg_tokens = estimate_tokens(str(msg["content"]))
            if total_tokens + msg_tokens < token_limit:
                valid_context.append(msg)
                valid_timestamps.append(ts)
                total_tokens += msg_tokens
    st.session_state.conversation_context = valid_context[-20:]
    st.session_state.context_timestamps = valid_timestamps[-20:]
    st.session_state.chat_history = st.session_state.chat_history[-20:]

    global status_text, progress_bar
    status_text, progress_bar = st.empty(), st.progress(0)

    try:
        update_progress("初始化查詢", 0.05)
        st.session_state.chat_history.append({"question": user_query, "answer": ""})
        st.session_state.last_user_query = user_query

        update_progress("分析問題意圖", 0.15)
        analysis = await analyze_and_screen(user_query, selected_cat, source_id, source_type, st.session_state.conversation_context)
        logger.info(f"分析完成：意圖={[i['intent'] for i in analysis.get('intents', [])]}")

        result, progress = await fetch_thread_data(user_query, selected_cat, source_id, source_type, analysis, st.session_state.conversation_context)
        update_progress("數據處理完成", progress, source_type)

        if result.get("rate_limit_until", 0) > time.time():
            wait_time = result["rate_limit_until"] - time.time()
            error_message = f"已達速率限制，請在 {wait_time:.1f} 秒後重試。"
            logger.warning(error_message)
            with st.chat_message("assistant"):
                st.error(error_message)
            st.session_state.chat_history[-1]["answer"] = error_message
            update_progress("速率限制", 1.0)
            status_text.empty()
            progress_bar.empty()
            st.session_state.awaiting_response = False
            return

        response = ""
        with st.chat_message("assistant"):
            col1, col2 = st.columns([0.95, 0.05])
            with col1:
                grok_container = st.empty()
            with col2:
                copy_container = st.empty()
            update_progress("正在生成回應", 0.85)
            logger.info(f"開始 stream_grok3_response，查詢：{user_query}")
            async for chunk in stream_grok3_response(
                user_query=user_query,
                metadata=[{"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0), "last_reply_time": item.get("last_reply_time", "0"), "like_count": item.get("like_count", 0), "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0} for item in result.get("thread_data", [])],
                thread_data={item["thread_id"]: item for item in result.get("thread_data", [])},
                processing=analysis.get("processing", {"intents": ["general_query"]}),
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
                logger.warning(f"查詢無回應：{user_query}")
                response = "無法生成回應，請稍後重試。"
                grok_container.markdown(response)
            copy_container.empty()
            render_copy_button(response, key=f"copy_new_{len(st.session_state.chat_history)}")

        st.session_state.chat_history[-1]["answer"] = response
        st.session_state.conversation_context.extend([
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response}
        ])
        st.session_state.context_timestamps.extend([time.time()] * 2)
        st.session_state.conversation_context = st.session_state.conversation_context[-20:]
        st.session_state.context_timestamps = st.session_state.context_timestamps[-20:]
        update_progress("完成", 1.0)

    except Exception as e:
        error_message = f"處理失敗：{str(e)}"
        logger.error(f"查詢處理錯誤：{user_query}，錯誤：{str(e)}")
        with st.chat_message("assistant"):
            st.error(error_message)
        st.session_state.chat_history[-1]["answer"] = error_message
        update_progress("處理失敗", 1.0)
    finally:
        status_text.empty()
        progress_bar.empty()
        st.session_state.awaiting_response = False

if __name__ == "__main__":
    asyncio.run(main())
