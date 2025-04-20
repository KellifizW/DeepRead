"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題，識別細粒度意圖，動態設置篩選條件。
- stream_grok3_response：生成流式回應，根據意圖動態選擇模板。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
"""

import aiohttp
import asyncio
import json
import re
import random
import math
import time
import logging
import traceback
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(function)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# Grok 3 API 配置
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000
API_TIMEOUT = 30  # 秒

def clean_html(text):
    """
    清理 HTML 標籤，規範化文本。
    Args:
        text (str): 包含 HTML 的文本。
    Returns:
        str: 清理後的純文本。
    """
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except Exception as e:
        logger.error(f"HTML cleaning failed: {str(e)}", extra={"function": "clean_html"})
        return text

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False, conversation_context=None):
    """
    分析用戶問題，識別細粒度意圖，動態設置篩選條件。
    Args:
        user_query (str): 用戶問題。
        cat_name (str): 分類名稱。
        cat_id (str): 分類 ID。
        thread_titles (list): 帖子標題列表。
        metadata (list): 帖子元數據。
        thread_data (dict): 帖子回覆數據。
        is_advanced (bool): 是否進行進階分析。
        conversation_context (list): 對話歷史。
    Returns:
        dict: 分析結果，包含意圖、主題、篩選參數、進階分析需求等。
    """
    conversation_context = conversation_context or []
    prompt = f"""
    你是 LIHKG 論壇的集體意見代表，根據用戶問題和提供的數據，以繁體中文回覆，模擬論壇用戶的語氣（例如吹水台輕鬆幽默，財經台專業嚴謹）。輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else ''}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else ''}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else ''}

    步驟：
    1. 判斷問題意圖並分類，根據語義和上下文，選擇以下之一：
       - list_titles：列出帖子標題。
         - 設置 direct_response=True，category_ids=[cat_id]，needs_advanced_analysis=False，reason="僅需標題列表"，data_type="title"，processing="list"。
         - 默認 filters：min_replies=0，min_likes=0。
         - 若問題包含「熱門」「高人氣」，設置 filters：min_replies=20，min_likes=10。
         - 若問題包含「搞笑」「感動」「財經」，設置 theme 並調整 filters（搞笑：min_likes=10；感動：min_likes=5；財經：min_likes=10）。
         - 示例：「列出帖文標題」「有哪些帖子？」「列出熱門標題」「列出搞笑帖子標題」
       - list_data：查詢可抓取數據類型。
         - 設置 direct_response=True，category_ids=[]，needs_advanced_analysis=False，reason="無需帖子分析"，data_type="none"，processing="list"。
         - 示例：「抓取什麼數據？」「提供什麼資料？」「能獲取哪些論壇信息？」
       - summarize_posts：總結帖子內容。
         - 設置 direct_response=False，category_ids=[cat_id]，繼續分析，processing="summarize"。
         - 設置 filters：min_replies=20，min_likes=10。
         - 示例：「分析吹水台搞笑帖子」「總結 LIHKG 熱門討論」
       - analyze_sentiment：分析帖子情緒。
         - 設置 direct_response=False，category_ids=[cat_id]，繼續分析，processing="sentiment"。
         - 設置 filters：min_replies=20，min_likes=10。
         - 示例：「吹水台帖子情緒如何？」「討論區的情緒分佈？」
       - self_introduction：自我介紹。
         - 設置 direct_response=True，category_ids=[]，needs_advanced_analysis=False，reason="無需帖子分析"，data_type="none"，processing="introduce"。
         - 示例：「你是誰？」「介紹一下 Grok」
       - general_query：無關 LIHKG 帖子。
         - 設置 direct_response=True，category_ids=[]，needs_advanced_analysis=False，reason="無關LIHKG問題"，data_type="none"，processing="general"。
         - 示例：「香港天氣如何？」「1+1 等於多少？」
    2. 若 direct_response=False，識別主題（感動、搞笑、財經等），標記為 theme，根據問題和分類推斷。
    3. 確定帖子數量（post_limit，1-10）：
       - 廣泛問題（例如「有哪些搞笑話題」）需要更多帖子（5-10）。
       - 具體問題（例如「某事件討論」）需要較少帖子（1-3）。
       - 參考對話歷史調整數量。
    4. 若 intent in ["summarize_posts", "analyze_sentiment"] 且提供 thread_data，檢查帖子是否達動態閾值：
       - 熱度標準：
         - 高熱度（like_count≥500 或 no_of_reply≥500）：閾值60%。
         - 中熱度（100≤like_count<500 或 100≤no_of_reply<500）：閾值40%。
         - 低熱度（其他）：閾值20%。
       - 對每篇帖子，計算總頁數（total_pages = ceil(no_of_reply/25)），檢查 fetched_pages 是否達 target_pages（ceil(total_pages * 閾值））。
       - 若任一帖子未達標，設置 needs_advanced_analysis=True，reason 記錄未達標帖子（如「帖子 X 僅抓取 Y/Z 頁，未達W%」）。
    5. 篩選帖子：
       - 若 intent="list_titles" 或 direct_response=False，設置初始抓取（30-90個標題）。
       - 從標題選10個候選（candidate_thread_ids），再選top_thread_ids。
    6. 設置參數：
       - direct_response：是否直接回應。
       - intent：問題意圖。
       - theme：問題主題（默認為空）。
       - category_ids：分類ID列表。
       - data_type："title"、"replies"、"both" 或 "none"。
       - post_limit：建議的帖子數量（默認5）。
       - reply_limit：{200 if is_advanced else 75}。
       - filters：根據意圖和主題設置。
       - processing：list、summarize、sentiment、introduce、general。
       - candidate_thread_ids：10個候選ID。
       - top_thread_ids：最終選定ID（不多於 post_limit）。
       - needs_advanced_analysis：是否需要進階分析。
       - reason：進階分析的原因。

    輸出：
    {{\"direct_response\": false, \"intent\": \"\", \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 5, \"reply_limit\": 0, \"filters\": {{}}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [], \"needs_advanced_analysis\": false, \"reason\": \"\"}}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "analyze_and_screen"})
        return {
            "direct_response": False,
            "intent": "unknown",
            "theme": "",
            "category_ids": [cat_id],
            "data_type": "both",
            "post_limit": 5,
            "reply_limit": 75,
            "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "Missing API key"
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你代表 LIHKG 論壇的集體意見，以繁體中文回答，僅基於提供數據，語氣適配分類。若問題無關LIHKG，則提供自然親切的直接回應。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.7
    }
    
    start_time = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                result["category_ids"] = [cat_id] if result.get("intent") == "list_titles" or not result.get("direct_response", False) else []
                result["intent"] = result.get("intent", "unknown")
                result["needs_advanced_analysis"] = result.get("needs_advanced_analysis", False)
                result["reason"] = result.get("reason", "")
                result["post_limit"] = result.get("post_limit", 5)
                result["filters"] = result.get("filters", {"min_replies": 0, "min_likes": 0})
                response_length = len(data["choices"][0]["message"]["content"])
                duration = time.time() - start_time
                logger.info(
                    json.dumps({
                        "event": "grok3_api_call",
                        "function": "analyze_and_screen",
                        "query": user_query,
                        "payload": payload,
                        "status": "success",
                        "response_length": response_length,
                        "duration_seconds": duration,
                        "intent": result["intent"],
                        "needs_advanced_analysis": result["needs_advanced_analysis"],
                        "filters": result["filters"]
                    }, ensure_ascii=False),
                    extra={"function": "analyze_and_screen"}
                )
                return result
    except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
        response_length = 0
        duration = time.time() - start_time
        logger.error(
            json.dumps({
                "event": "grok3_api_call",
                "function": "analyze_and_screen",
                "query": user_query,
                "payload": payload,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e),
                "stack_trace": traceback.format_exc(),
                "response_length": response_length,
                "duration_seconds": duration
            }, ensure_ascii=False),
            extra={"function": "analyze_and_screen"}
        )
        return {
            "direct_response": True,
            "intent": "general_query",
            "theme": "",
            "category_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "reply_limit": 75,
            "filters": {"min_replies": 0, "min_likes": 0},
            "processing": "general",
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": f"Analysis failed: {str(e)}"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None):
    """
    使用 Grok 3 API 生成流式回應，根據意圖動態選擇模板。
    Args:
        user_query (str): 用戶問題。
        metadata (list): 帖子元數據。
        thread_data (dict): 帖子回覆數據。
        processing (str): 處理類型（list、summarize、sentiment 等）。
        conversation_context (list): 對話歷史。
        needs_advanced_analysis (bool): 是否需要進階分析。
        reason (str): 進階分析原因。
        filters (dict): 篩選條件。
    Yields:
        str: 回應片段。
    """
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 0, "min_likes": 0}
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"Grok 3 API key missing: {str(e)}", extra={"function": "stream_grok3_response"})
        yield "錯誤: 缺少 API 密鑰"
        return
    
    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"],
            "title": data["title"],
            "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0),
            "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [r for r in data.get("replies", []) if r.get("like_count", 0) >= 5][:25],
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }
    
    prompt_templates = {
        "list": f"""
        你是 LIHKG 論壇的數據助手，以繁體中文回答，語氣簡潔清晰。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子元數據：{json.dumps(metadata, ensure_ascii=False)}
        篩選條件：{json.dumps(filters, ensure_ascii=False)}
        僅列出帖子標題，格式為：
        - 帖子 ID: [thread_id] 標題: [title]
        若無符合條件的帖子，回答：「無符合條件的帖子（篩選：回覆數≥{filters.get('min_replies', 0)}，點讚數≥{filters.get('min_likes', 0)}）。」並列出最多5個最新帖子標題（無篩選）。
        若無任何帖子，回答：「目前無可用帖子標題。」
        輸出：標題列表
        """,
        "summarize": f"""
        你是 LIHKG 論壇的集體意見代表，總結帖子內容，300-500字。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        引用高關注回覆（like_count≥5），語氣適配分類（吹水台輕鬆，財經台專業）。
        輸出：總結
        """,
        "sentiment": f"""
        你是 LIHKG 論壇的集體意見代表，分析帖子情緒。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        判斷情緒分佈（正面、負面、中立），聚焦高關注回覆（like_count≥5）。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...
        """,
        "introduce": f"""
        你是 Grok 3，由 xAI 創建，以繁體中文回答，語氣自然親切。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        回答：「我是 Grok 3，由 xAI 創建，專為解答您的問題而生！有什麼可以幫您？」（50-100字）。
        輸出：自我介紹
        """,
        "general": f"""
        你是 Grok 3，由 xAI 創建，以繁體中文回答，語氣自然親切。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        提供簡潔直接的回應，聚焦問題核心。
        輸出：直接回應
        """
    }
    
    # 非 LIHKG 問題或無帖子數據的提示
    if not metadata and not filtered_thread_data:
        prompt = f"""
        你是 Grok 3，由 xAI 創建，以繁體中文回答，語氣自然親切。問題：{user_query}
        對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
        若問題要求列出數據類型（如「抓取什麼數據」「提供什麼資料」），回答：「我可以抓取 LIHKG 討論區的以下數據：帖子標題（title）、帖子 ID（thread_id）、回覆數量（no_of_reply）、最後回覆時間（last_reply_time）、點贊數（like_count）、踩數（dislike_count），以及部分回覆的內容（msg）、回覆的點贊與踩數、回覆時間（reply_time）。此外，還包括抓取頁數（fetched_pages）資訊，以評估數據完整性。若需具體帖子分析，請提供更多細節！」（100-150字）。
        若問題為自我介紹（如「你是誰？」），回答：「我是 Grok 3，由 xAI 創建，專為解答您的問題而生！有什麼可以幫您？」（50-100字）。
        其他非 LIHKG 問題，提供簡潔直接的回應。
        若無帖子數據且問題要求帖子（如「列出標題」），回答：「目前無可用帖子標題。」
        輸出：直接回應
        """
    else:
        prompt = prompt_templates.get(processing, prompt_templates["summarize"])
    
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: {len(prompt)} characters", extra={"function": "stream_grok3_response"})
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    messages = [
        {"role": "system", "content": "你代表 LIHKG 論壇的集體意見或作為Grok 3直接回答，以繁體中文回答，語氣適配分類或自然親切。"},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3-beta",
        "messages": messages,
        "max_tokens": 2000,
        "temperature": 0.7,
        "stream": True
    }
    
    start_time = time.time()
    response_content = ""
    async with aiohttp.ClientSession() as session:
        try:
            for attempt in range(3):
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        async for line in response.content:
                            if line and not line.isspace():
                                line_str = line.decode('utf-8').strip()
                                if line_str == "data: [DONE]":
                                    break
                                if line_str.startswith("data: "):
                                    try:
                                        chunk = json.loads(line_str[6:])
                                        content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                        if content:
                                            if "###" in content:
                                                logger.warning(f"Detected ### in response, retrying with simplified prompt", extra={"function": "stream_grok3_response"})
                                                raise ValueError("Content moderation detected")
                                            response_content += content
                                            yield content
                                    except json.JSONDecodeError as e:
                                        logger.warning(f"JSON decode error in stream chunk: {str(e)}", extra={"function": "stream_grok3_response"})
                                        continue
                        # 附加進階分析建議
                        if needs_advanced_analysis and metadata and filtered_thread_data:
                            yield f"\n建議：為確保分析全面，建議抓取更多帖子頁數。{reason}\n"
                        duration = time.time() - start_time
                        logger.info(
                            json.dumps({
                                "event": "grok3_api_call",
                                "function": "stream_grok3_response",
                                "query": user_query,
                                "payload": payload,
                                "status": "success",
                                "response_length": len(response_content),
                                "duration_seconds": duration
                            }, ensure_ascii=False),
                            extra={"function": "stream_grok3_response"}
                        )
                        return
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(f"Grok 3 request failed, attempt {attempt+1}: {str(e)}", extra={"function": "stream_grok3_response"})
                    if attempt < 2:
                        for tid in filtered_thread_data:
                            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:5]
                        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
                        payload["messages"][-1]["content"] = prompt
                        await asyncio.sleep(2 + attempt * 2)
                        continue
                    raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                json.dumps({
                    "event": "grok3_api_call",
                    "function": "stream_grok3_response",
                    "query": user_query,
                    "payload": payload,
                    "status": "failed",
                    "error_type": type(e).__name__,
                    "error": str(e),
                    "stack_trace": traceback.format_exc(),
                    "response_length": len(response_content),
                    "duration_seconds": duration
                }, ensure_ascii=False),
                extra={"function": "stream_grok3_response"}
            )
            yield f"錯誤：生成回應失敗（{str(e)}），請稍後重試"
        finally:
            await session.close()

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None, conversation_context=None):
    """
    處理用戶問題，抓取並分析 LIHKG 帖子。
    Args:
        user_question (str): 用戶問題。
        selected_cat (str): 分類名稱。
        cat_id (str): 分類 ID。
        analysis (dict): 問題分析結果。
        request_counter (int): 請求計數。
        last_reset (float): 最後重置時間。
        rate_limit_until (float): 速率限制解除時間。
        is_advanced (bool): 是否進階分析。
        previous_thread_ids (list): 前次帖子 ID。
        previous_thread_data (dict): 前次帖子數據。
        conversation_context (list): 對話歷史。
    Returns:
        dict: 處理結果，包含帖子數據、速率限制信息等。
    """
    try:
        # 進行問題分析與意圖識別
        analysis = await analyze_and_screen(
            user_query=user_question,
            cat_name=selected_cat,
            cat_id=cat_id,
            thread_titles=None,
            metadata=None,
            thread_data=previous_thread_data,
            is_advanced=is_advanced,
            conversation_context=conversation_context
        )
        
        # 檢查速率限制
        if rate_limit_until > time.time():
            logger.warning(f"Rate limit active until {rate_limit_until}", extra={"function": "process_user_question"})
            return {
                "selected_cat": selected_cat,
                "thread_data": [],
                "rate_limit_info": [{"message": "Rate limit active", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        # 若 direct_response=True 或僅需標題，限制抓取
        if analysis.get("direct_response", True) or analysis.get("intent") == "list_titles":
            thread_data = []
            rate_limit_info = []
            if analysis.get("intent") == "list_titles":
                initial_threads = []
                for page in range(1, 4):
                    result = await get_lihkg_topic_list(
                        cat_id=cat_id,
                        start_page=page,
                        max_pages=1,
                        request_counter=request_counter,
                        last_reset=last_reset,
                        rate_limit_until=rate_limit_until
                    )
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    initial_threads.extend(result.get("items", []))
                    logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}", extra={"function": "process_user_question"})
                    if len(initial_threads) >= 90:
                        initial_threads = initial_threads[:90]
                        break
                
                filters = analysis.get("filters", {"min_replies": 0, "min_likes": 0})
                min_replies = filters.get("min_replies", 0)
                min_likes = filters.get("min_likes", 0)
                previous_thread_ids = previous_thread_ids or []
                
                filtered_items = [
                    item for item in initial_threads
                    if item.get("no_of_reply", 0) >= min_replies
                    and int(item.get("like_count", 0)) >= min_likes
                    and str(item["thread_id"]) not in previous_thread_ids
                ]
                
                post_limit = min(analysis.get("post_limit", 5), 10)
                if not filtered_items:
                    logger.warning(
                        f"No threads meet filters: min_replies={min_replies}, min_likes={min_likes}, trying without filters",
                        extra={"function": "process_user_question"}
                    )
                    filtered_items = [
                        item for item in initial_threads
                        if str(item["thread_id"]) not in previous_thread_ids
                    ]
                
                thread_data = [
                    {
                        "thread_id": str(item["thread_id"]),
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    } for item in filtered_items[:post_limit]
                ]
                for item in thread_data:
                    thread_id = item["thread_id"]
                    if thread_id not in st.session_state.thread_cache:
                        st.session_state.thread_cache[thread_id] = {
                            "data": item,
                            "timestamp": time.time()
                        }
                logger.info(
                    json.dumps({
                        "event": "thread_preparation",
                        "function": "process_user_question",
                        "query": user_question,
                        "initial_threads": len(initial_threads),
                        "filtered_items": len(filtered_items),
                        "thread_data": len(thread_data),
                        "filters": filters
                    }, ensure_ascii=False),
                    extra={"function": "process_user_question"}
                )
            
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        post_limit = min(analysis.get("post_limit", 3), 10)
        reply_limit = 200 if is_advanced else min(analysis.get("reply_limit", 75), 75)
        filters = analysis.get("filters", {"min_replies": 20, "min_likes": 10})
        min_replies = filters.get("min_replies", 20)
        min_likes = filters.get("min_likes", 10)
        candidate_thread_ids = analysis.get("candidate_thread_ids", [])
        top_thread_ids = analysis.get("top_thread_ids", []) if not is_advanced else []
        
        thread_data = []
        rate_limit_info = []
        previous_thread_ids = previous_thread_ids or []
        
        if is_advanced and previous_thread_ids:
            for thread_id in previous_thread_ids:
                cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
                if not cached_data:
                    continue
                fetched_pages = cached_data.get("fetched_pages", [])
                existing_replies = cached_data.get("replies", [])
                total_replies = cached_data.get("no_of_reply", 0)
                
                like_count = cached_data.get("like_count", 0)
                no_of_reply = total_replies
                if like_count >= 500 or no_of_reply >= 500:
                    threshold = 0.6
                elif like_count >= 100 or no_of_reply >= 100:
                    threshold = 0.4
                else:
                    threshold = 0.2
                total_pages = (total_replies + 24) // 25
                target_pages = math.ceil(total_pages * threshold)
                remaining_pages = max(0, target_pages - len(fetched_pages))
                
                if remaining_pages <= 0:
                    logger.info(f"Thread {thread_id} meets {int(threshold*100)}% page threshold: {len(fetched_pages)}/{target_pages}", extra={"function": "process_user_question"})
                    thread_data.append({
                        "thread_id": str(thread_id),
                        "title": cached_data.get("title", "未知標題"),
                        "no_of_reply": total_replies,
                        "last_reply_time": cached_data.get("last_reply_time", 0),
                        "like_count": cached_data.get("like_count", 0),
                        "dislike_count": cached_data.get("dislike_count", 0),
                        "replies": existing_replies,
                        "fetched_pages": fetched_pages
                    })
                    continue
                
                start_page = max(fetched_pages, default=1) + 1 if fetched_pages else 1
                thread_result = await get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    request_counter=request_counter,
                    last_reset=last_reset,
                    rate_limit_until=rate_limit_until,
                    max_replies=reply_limit,
                    fetch_last_pages=remaining_pages,
                    start_page=start_page
                )
                
                request_counter = thread_result.get("request_counter", request_counter)
                last_reset = thread_result.get("last_reset", last_reset)
                rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(thread_result.get("rate_limit_info", []))
                
                replies = thread_result.get("replies", [])
                if not replies and thread_result.get("total_replies", 0) >= min_replies:
                    logger.warning(f"Invalid thread: {thread_id}", extra={"function": "process_user_question"})
                    continue
                
                all_replies = existing_replies + [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in replies]
                sorted_replies = sorted(all_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
                all_fetched_pages = sorted(set(fetched_pages + thread_result.get("fetched_pages", [])))
                
                thread_data.append({
                    "thread_id": str(thread_id),
                    "title": thread_result.get("title", cached_data.get("title", "未知標題")),
                    "no_of_reply": thread_result.get("total_replies", total_replies),
                    "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0)),
                    "like_count": thread_result.get("like_count", cached_data.get("like_count", 0)),
                    "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0)),
                    "replies": sorted_replies,
                    "fetched_pages": all_fetched_pages
                })
                logger.info(f"Advanced thread {thread_id}: replies={len(sorted_replies)}, pages={len(all_fetched_pages)}/{target_pages}", extra={"function": "process_user_question"})
                await asyncio.sleep(1)
            
            logger.info(f"Advanced processing completed: {len(thread_data)} threads", extra={"function": "process_user_question"})
            return {
                "selected_cat": selected_cat,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        initial_threads = []
        for page in range(1, 4):
            result = await get_lihkg_topic_list(
                cat_id=cat_id,
                start_page=page,
                max_pages=1,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until
            )
            request_counter = result.get("request_counter", request_counter)
            last_reset = result.get("last_reset", last_reset)
            rate_limit_until = result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(result.get("rate_limit_info", []))
            initial_threads.extend(result.get("items", []))
            logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}", extra={"function": "process_user_question"})
            if len(initial_threads) >= 90:
                initial_threads = initial_threads[:90]
                break
        
        filters = analysis.get("filters", {"min_replies": 20, "min_likes": 10})
        min_replies = filters.get("min_replies", 20)
        min_likes = filters.get("min_likes", 10)
        filtered_items = [
            item for item in initial_threads
            if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
            and str(item["thread_id"]) not in previous_thread_ids
        ]
        logger.info(
            json.dumps({
                "event": "thread_filtering",
                "function": "process_user_question",
                "initial_threads": len(initial_threads),
                "filtered_items": len(filtered_items),
                "filters": filters,
                "excluded_thread_ids": previous_thread_ids
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        
        for item in initial_threads:
            thread_id = str(item["thread_id"])
            if thread_id not in st.session_state.thread_cache:
                st.session_state.thread_cache[thread_id] = {
                    "data": {
                        "thread_id": thread_id,
                        "title": item["title"],
                        "no_of_reply": item.get("no_of_reply", 0),
                        "last_reply_time": item.get("last_reply_time", 0),
                        "like_count": item.get("like_count", 0),
                        "dislike_count": item.get("dislike_count", 0),
                        "replies": [],
                        "fetched_pages": []
                    },
                    "timestamp": time.time()
                }
        
        analysis = await analyze_and_screen(
            user_query=user_question,
            cat_name=selected_cat,
            cat_id=cat_id,
            thread_titles=[item["title"] for item in filtered_items[:90]],
            metadata=None,
            thread_data=None,
            conversation_context=conversation_context
        )
        top_thread_ids = analysis.get("top_thread_ids", [])
        if not top_thread_ids and filtered_items:
            top_thread_ids = [item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]
            logger.warning(f"No top_thread_ids, randomly selected: {top_thread_ids}", extra={"function": "process_user_question"})
        
        candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not candidate_threads:
            candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items))) if filtered_items else []
            logger.info(f"No candidate threads, using random: {len(candidate_threads)}", extra={"function": "process_user_question"})
        
        for item in candidate_threads:
            thread_id = str(item["thread_id"])
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=25,
                fetch_last_pages=0
            )
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"Invalid thread: {thread_id}", extra={"function": "process_user_question"})
                continue
            
            sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:25]
            thread_data.append({
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            })
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[-1]["replies"],
                "fetched_pages": thread_data[-1]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            logger.info(f"Fetched candidate thread {thread_id}: replies={len(replies)}", extra={"function": "process_user_question"})
            await asyncio.sleep(1)
        
        final_threads = [item for item in filtered_items if str(item["thread_id"]) in map(str, top_thread_ids)][:post_limit]
        if not final_threads:
            final_threads = candidate_threads[:post_limit]
        
        for item in final_threads:
            thread_id = str(item["thread_id"])
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                request_counter=request_counter,
                last_reset=last_reset,
                rate_limit_until=rate_limit_until,
                max_replies=reply_limit,
                fetch_last_pages=2
            )
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"Invalid thread: {thread_id}", extra={"function": "process_user_question"})
                continue
            
            sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            thread_data.append({
                "thread_id": thread_id,
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0),
                "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0),
                "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
                "fetched_pages": thread_result.get("fetched_pages", [1])
            })
            st.session_state.thread_cache[thread_id]["data"].update({
                "replies": thread_data[-1]["replies"],
                "fetched_pages": thread_data[-1]["fetched_pages"]
            })
            st.session_state.thread_cache[thread_id]["timestamp"] = time.time()
            logger.info(f"Fetched final thread {thread_id}: replies={len(replies)}", extra={"function": "process_user_question"})
            await asyncio.sleep(1)
        
        logger.info(f"Processing completed: {len(thread_data)} threads for query: {user_question}", extra={"function": "process_user_question"})
        return {
            "selected_cat": selected_cat,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(
            json.dumps({
                "event": "processing_error",
                "function": "process_user_question",
                "query": user_question,
                "status": "failed",
                "error_type": type(e).__name__,
                "error": str(e),
                "stack_trace": traceback.format_exc()
            }, ensure_ascii=False),
            extra={"function": "process_user_question"}
        )
        return {
            "selected_cat": selected_cat,
            "thread_data": [],
            "rate_limit_info": [{"message": f"Processing failed: {str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis or {}
        }