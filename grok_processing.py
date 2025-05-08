import aiohttp
import asyncio
import json
import re
import datetime
import time
import logging
import streamlit as st
import pytz
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content
from reddit_api import get_reddit_topic_list, get_reddit_thread_content
from logging_config import configure_logger
from dynamic_prompt_utils import build_dynamic_prompt, parse_query, extract_keywords, CONFIG
import traceback

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 120000
API_TIMEOUT = 120  # Increased timeout for large prompts
MAX_CACHE_SIZE = 100

cache_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

def clean_html(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r'<[^>]+>')
        text = clean.sub('', text)
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            return "[表情符號]" if "hkgmoji" in text else "[圖片]" if any(ext in text.lower() for ext in ['.webp', '.jpg', '.png']) else "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML 清理失敗：{str(e)}")
        return text

def clean_response(response):
    if isinstance(response, str):
        cleaned = re.sub(r'\[post_id: [a-f0-9]{40}\]', '[回覆]', response)
        return cleaned
    return response

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "一般", "keywords": []}
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"theme": "一般", "keywords": []}
    
    prompt = f"""
你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
特別注意用戶問題中的意圖（例如「熱門」「總結」「追問」）和回應中的帖子標題。
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
輸出格式：{{"theme": "主要主題", "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]}}
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                status_code = response.status
                if status_code != 200:
                    logger.warning(f"對話摘要失敗：狀態碼={status_code}")
                    return {"theme": "一般", "keywords": []}
                data = await response.json()
                usage = data.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                response_content = data["choices"][0]["message"]["content"]
                logger.info(
                    f"Grok3 API 調用：函數=summarize_context, "
                    f"狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                    f"輸出 token={completion_tokens}, 回應={response_content}, "
                    f"提示長度={len(prompt)} 字符"
                )
                result = json.loads(response_content)
                return result
    except Exception as e:
        logger.warning(f"對話摘要錯誤：{str(e)}")
        return {"theme": "一般", "keywords": []}

async def analyze_and_screen(user_query, source_name, source_id, source_type="lihkg", conversation_context=None):
    conversation_context = conversation_context or []
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {
            "direct_response": True,
            "intents": [{"intent": "general_query", "confidence": 0.5, "reason": "缺少 API 密鑰"}],
            "theme": "一般",
            "source_type": source_type,
            "source_ids": [],
            "data_type": "none",
            "post_limit": 5,
            "filters": {},
            "processing": {"intents": ["general_query"]},
            "candidate_thread_ids": [],
            "top_thread_ids": [],
            "needs_advanced_analysis": False,
            "reason": "缺少 API 密鑰",
            "theme_keywords": []
        }
    
    logger.info(f"開始語義分析：查詢={user_query}")
    parsed_query = await parse_query(user_query, conversation_context, GROK3_API_KEY, source_type)
    intents = parsed_query["intents"]
    query_keywords = parsed_query["keywords"]
    top_thread_ids = parsed_query["thread_ids"]
    reason = parsed_query["reason"]
    confidence = parsed_query["confidence"]
    
    if not intents:
        logger.warning(f"意圖分析失敗：查詢={user_query}, 原因=無法識別有效意圖，回退到默認意圖 recommend_threads")
        intents = [{"intent": "recommend_threads", "confidence": 0.5, "reason": "無法識別有效意圖，回退到默認意圖"}]
        reason = "無法識別有效意圖，回退到默認意圖"
        confidence = 0.5
    
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "一般")
    historical_keywords = context_summary.get("keywords", [])
    
    has_high_confidence_list_titles = any(i["intent"] == "list_titles" and i["confidence"] >= 0.9 for i in intents)
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"]) and not has_high_confidence_list_titles
    
    if is_vague and historical_theme != "一般":
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": f"問題模糊，延續歷史主題：{historical_theme}"}]
        reason = f"問題模糊，延續歷史主題：{historical_theme}"
    elif is_vague:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "問題模糊，默認總結帖子"}]
        reason = "問題模糊，默認總結帖子"
    
    theme = historical_theme if is_vague else (query_keywords[0] if query_keywords else "一般")
    theme_keywords = historical_keywords if is_vague else query_keywords
    
    post_limit = 15 if any(i["intent"] == "list_titles" for i in intents) else (20 if any(i["intent"] in ["search_keywords", "find_themed"] for i in intents) else 5)
    data_type = "both" if not all(i["intent"] in ["general_query", "introduce"] for i in intents) else "none"
    
    if any(i["intent"] == "follow_up" for i in intents):
        post_limit = len(top_thread_ids) or 2
        data_type = "replies"
    
    logger.info(f"語義分析結果：intents={[i['intent'] for i in intents]}, confidence={confidence}, reason={reason}")
    return {
        "direct_response": all(i["intent"] in ["general_query", "introduce"] for i in intents),
        "intents": intents,
        "theme": theme,
        "source_type": source_type,
        "source_ids": [source_id],
        "data_type": data_type,
        "post_limit": post_limit,
        "filters": {"min_replies": 10, "min_likes": 0, "sort": "popular", "keywords": theme_keywords},
        "processing": {"intents": [i["intent"] for i in intents], "top_thread_ids": top_thread_ids, "analysis": parsed_query},
        "candidate_thread_ids": top_thread_ids,
        "top_thread_ids": top_thread_ids,
        "needs_advanced_analysis": confidence < 0.7,
        "reason": reason,
        "theme_keywords": theme_keywords
    }

async def prioritize_threads_with_grok(user_query, threads, source_name, source_id, source_type="lihkg", intents=["summarize_posts"]):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        return {"top_thread_ids": [], "reason": "缺少 API 密鑰", "intent_breakdown": []}

    if any(intent == "follow_up" for intent in intents):
        referenced_thread_ids = []
        context = st.session_state.get("conversation_context", [])
        if context:
            last_response = context[-1].get("content", "")
            matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
            referenced_thread_ids = [tid for tid in matches if any(str(t["thread_id"]) == tid for t in threads)]
        if referenced_thread_ids:
            logger.info(f"追問檢測到參考帖子：thread_ids={referenced_thread_ids}")
            return {
                "top_thread_ids": referenced_thread_ids[:5],
                "reason": "使用追問的參考帖子 ID",
                "intent_breakdown": [{"intent": "follow_up", "thread_ids": referenced_thread_ids[:5]}]
            }

    threads = [{"thread_id": str(t["thread_id"]), **t} for t in threads]
    prompt = f"""
你是帖子優先級排序助手，請根據用戶查詢和多個意圖，從提供的帖子中選出最多20個最相關的帖子。
查詢：{user_query}
意圖：{json.dumps(intents, ensure_ascii=False)}
討論區：{source_name} (ID: {source_id})
來源類型：{source_type}
帖子數據：
{json.dumps([{"thread_id": str(t["thread_id"]), "title": clean_html(t["title"]), "no_of_reply": t.get("no_of_reply", 0), "like_count": t.get("like_count", 0)} for t in threads], ensure_ascii=False)}
輸出格式：{{
  "top_thread_ids": ["id1", "id2", ...],
  "reason": "排序原因",
  "intent_breakdown": [
    {{"intent": "意圖1", "thread_ids": ["id1", "id2"]}},
    ...
  ]
}}
請確保 reason 說明簡潔，不超過50字。
"""
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 500,
        "temperature": 0.7
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.debug(f"帖子優先級排序失敗：狀態碼={status_code}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"帖子優先級排序失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", 0)
                    completion_tokens = usage.get("completion_tokens", 0)
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(
                        f"Grok3 API 調用：函數=prioritize_threads_with_grok, "
                        f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                        f"輸出 token={completion_tokens}, 回應={response_content}, "
                        f"提示長度={len(prompt)} 字符"
                    )
                    try:
                        result = json.loads(response_content)
                        top_thread_ids = result.get("top_thread_ids", [])
                        reason = result.get("reason", "無排序原因")
                        intent_breakdown = result.get("intent_breakdown", [])
                        top_thread_ids = [str(tid) for tid in top_thread_ids]
                        thread_id_set = [str(t["thread_id"]) for t in threads]
                        valid_thread_ids = list(dict.fromkeys([tid for tid in top_thread_ids if tid in thread_id_set]))
                        invalid_thread_ids = [tid for tid in top_thread_ids if tid not in thread_id_set]
                        logger.info(
                            f"thread_id 驗證：有效 thread_ids 數量={len(valid_thread_ids)}, "
                            f"無效 thread_ids 數量={len(invalid_thread_ids)}"
                        )
                        logger.info(f"成功解析 JSON，回應包含 {len(valid_thread_ids)} 個有效 thread_ids")
                        return {
                            "top_thread_ids": valid_thread_ids,
                            "reason": reason,
                            "intent_breakdown": intent_breakdown
                        }
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"無法解析優先級排序結果：原始回應={response_content}, "
                            f"token 使用：輸入={prompt_tokens}, 輸出={completion_tokens}, 錯誤：{str(e)}"
                        )
                        try:
                            match = re.search(r'"top_thread_ids":\s*\[(.*?)\]', response_content, re.DOTALL)
                            if match:
                                ids_str = match.group(1)
                                ids = [str(id.strip().strip('"')) for id in ids_str.split(',') if id.strip()]
                                thread_id_set = [str(t["thread_id"]) for t in threads]
                                valid_thread_ids = list(dict.fromkeys([tid for tid in ids if tid in thread_id_set]))
                                invalid_thread_ids = [tid for tid in ids if tid not in thread_id_set]
                                logger.info(
                                    f"從不完整回應提取：有效 thread_ids 數量={len(valid_thread_ids)}, "
                                    f"無效 thread_ids 數量={len(invalid_thread_ids)}"
                                )
                                if valid_thread_ids:
                                    logger.info(f"成功從不完整回應中提取 {len(valid_thread_ids)} 個 thread_ids")
                                    return {
                                        "top_thread_ids": valid_thread_ids,
                                        "reason": "從不完整回應中提取的 thread_ids",
                                        "intent_breakdown": []
                                    }
                        except Exception as extract_e:
                            logger.warning(f"無法從不完整回應中提取 thread_ids：{str(extract_e)}")
                        return {"top_thread_ids": [], "reason": f"無法解析 API 回應：{str(e)}", "intent_breakdown": []}
        except Exception as e:
            logger.debug(f"帖子優先級排序錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2)
                continue
            sorted_threads = sorted(
                threads,
                key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                reverse=True
            )
            valid_thread_ids = [str(t["thread_id"]) for t in sorted_threads[:20]]
            logger.info(f"API 調用失敗，回退到熱門度排序，返回 {len(valid_thread_ids)} 個 thread_ids")
            return {
                "top_thread_ids": valid_thread_ids,
                "reason": f"優先級排序失敗（{str(e)}），回退到熱門度排序",
                "intent_breakdown": []
            }

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_source, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, source_id=None, source_type="lihkg"):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 10, "min_likes": 0}
    
    # 確保 selected_source 是字典格式
    if isinstance(selected_source, str):
        selected_source = {"source_name": selected_source, "source_type": source_type}
        logger.debug(f"將字符串 selected_source 轉換為字典：{selected_source}")
    elif not isinstance(selected_source, dict):
        logger.warning(f"無效的 selected_source 類型：{type(selected_source)}，使用默認值")
        selected_source = {"source_name": "未知", "source_type": source_type}
    elif "source_name" not in selected_source or "source_type" not in selected_source:
        logger.warning(f"selected_source 缺少必要字段：{selected_source}，補充默認值")
        selected_source = {
            "source_name": selected_source.get("source_name", "未知"),
            "source_type": selected_source.get("source_type", source_type)
        }
    
    if not thread_data:
        error_msg = f"在 {selected_source['source_name']} 中未找到符合條件的帖子（篩選：{json.dumps(filters, ensure_ascii=False)}）。建議嘗試其他關鍵詞或討論區！"
        logger.warning(f"無匹配帖子：{error_msg}")
        yield error_msg
        return

    if not isinstance(processing, dict):
        logger.error(f"無效的處理數據格式：預期 dict，得到 {type(processing)}")
        yield f"錯誤：無效的處理數據格式（{type(processing)}）。請聯繫支持。"
        return
    
    intents_info = []
    if processing.get('analysis') and processing['analysis'].get('intents'):
        intents_info = processing['analysis']['intents']
    elif processing.get('intents'):
        intents_info = [{"intent": i, "confidence": 0.7, "reason": "從 processing.intents 提取"} for i in processing['intents']]
    else:
        logger.warning(f"未找到有效意圖，回退到默認意圖：summarize_posts")
        intents_info = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "無有效意圖，默認總結"}]
    
    intents = [i['intent'] for i in intents_info]
    logger.info(f"Starting stream_grok3_response for query: {user_query}, intents: {intents}, source: {selected_source}")

    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError as e:
        logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
        yield "錯誤：缺少 API 密鑰"
        return
    
    total_min_tokens = 0
    total_max_tokens = 0
    for intent in intents:
        word_min, word_max = CONFIG["default_word_ranges"].get(intent, (500, 1500))
        total_min_tokens += int(word_min / 0.8)
        total_max_tokens += int(word_max / 0.8)
    
    prompt_length = len(json.dumps(thread_data, ensure_ascii=False)) + len(user_query) + 1000
    length_factor = min(prompt_length / GROK3_TOKEN_LIMIT, 1.0)
    target_tokens = total_min_tokens + (total_max_tokens - total_min_tokens) * length_factor
    target_tokens = min(max(int(target_tokens), total_min_tokens), total_max_tokens)

    max_tokens_limit = 8000
    max_tokens = min(target_tokens + 500, max_tokens_limit)
    max_replies_per_thread = 300 if any(intent == "follow_up" for intent in intents) else 100
    max_comments = 300 if source_type == "reddit" and any(intent == "follow_up" for intent in intents) else 100

    # Dynamically adjust max_replies_per_thread based on prompt size
    if prompt_length > GROK3_TOKEN_LIMIT * 0.8:
        max_replies_per_thread = max_replies_per_thread // 2
        logger.debug(f"Prompt length {prompt_length} exceeds 80% of limit, reducing max_replies_per_thread to {max_replies_per_thread}")

    if any(intent == "list_titles" for intent in intents):
        max_replies_per_thread = 10
    elif any(intent == "analyze_sentiment" for intent in intents):
        max_replies_per_thread = 200

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    
    if any(intent in ["follow_up", "fetch_thread_by_id", "analyze_sentiment"] for intent in intents):
        reply_count_prompt = f"""
你是資料抓取助手，請根據問題和意圖決定每個帖子應下載的回覆數量（100、200、250、300、500 條）。
僅以 JSON 格式回應，禁止生成自然語言或其他格式的內容。
問題：{user_query}
意圖：{json.dumps(intents, ensure_ascii=False)}
若問題需要深入分析（如情緒分析、意見分類、追問、特定帖子ID），建議較多回覆（200-500）。
默認：100 條。
輸出格式：{{"replies_per_thread": 100, "reason": "決定原因"}}
"""
        payload = {
            "model": "grok-3",
            "messages": [{"role": "user", "content": reply_count_prompt}],
            "max_tokens": 100,
            "temperature": 0.5
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code == 200:
                        data = await response.json()
                        usage = data.get("usage", {})
                        prompt_tokens = usage.get("prompt_tokens", 0)
                        completion_tokens = usage.get("completion_tokens", 0)
                        response_content = data["choices"][0]["message"]["content"]
                        logger.info(
                            f"Grok3 API 調用：函數=stream_grok3_response (reply_count), "
                            f"查詢={user_query}, 狀態碼={status_code}, 輸入 token={prompt_tokens}, "
                            f"輸出 token={completion_tokens}, 回應={response_content}, "
                            f"提示長度={len(reply_count_prompt)} 字符"
                        )
                        result = json.loads(response_content)
                        max_replies_per_thread = min(result.get("replies_per_thread", 300 if any(intent == "follow_up" for intent in intents) else 200 if any(intent == "analyze_sentiment" for intent in intents) else 100), 500)
                        logger.debug(f"選擇每帖子回覆數：{max_replies_per_thread}, 原因：{result.get('reason', '未提供')}")
                    else:
                        logger.warning(f"無法確定每帖子回覆數，狀態碼={status_code}")
        except Exception as e:
            logger.warning(f"每帖子回覆數選擇失敗：{str(e)}")

    thread_data_dict = {}
    if isinstance(thread_data, list):
        thread_data_dict = {str(data["thread_id"]): data for data in thread_data if isinstance(data, dict) and "thread_id" in data}
    elif isinstance(thread_data, dict):
        thread_data_dict = thread_data
    else:
        logger.error(f"無效的 thread_data 格式：預期 list 或 dict，得到 {type(thread_data)}")
        yield f"錯誤：無效的 thread_data 格式（{type(thread_data)}）。請聯繫支持。"
        return
    
    if any(intent in ["follow_up", "fetch_thread_by_id"] for intent in intents):
        referenced_thread_ids = []
        if any(intent == "follow_up" for intent in intents):
            referenced_thread_ids = processing.get('analysis', {}).get('thread_ids', [])
            if not referenced_thread_ids:
                last_response = conversation_context[-1].get("content", "") if conversation_context else ""
                matches = re.findall(r"\[帖子 ID: (\d+)\]", last_response)
                referenced_thread_ids = [tid for tid in matches if any(str(t["thread_id"]) == tid for t in metadata)]
        else:
            top_thread_ids = processing.get("top_thread_ids", [])
            referenced_thread_ids = [tid for tid in top_thread_ids if str(tid) in thread_data_dict]
        
        if not referenced_thread_ids and any(intent == "follow_up" for intent in intents):
            keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
            theme_keywords = keyword_result["keywords"]
            sort = "new" if keyword_result.get("time_sensitive", False) else "best"
            logger.debug(f"追問無參考帖子，獲取補充帖子，排序：{sort}，關鍵詞：{theme_keywords}")
            async with request_semaphore:
                if source_type == "lihkg":
                    supplemental_result = await get_lihkg_topic_list(
                        cat_id=source_id,
                        start_page=1,
                        max_pages=2
                    )
                else:
                    supplemental_result = await get_reddit_topic_list(
                        subreddit=source_id,
                        start_page=1,
                        max_pages=2,
                        sort=sort
                    )
            supplemental_threads = supplemental_result.get("items", [])
            if not supplemental_threads:
                logger.warning(f"無法獲取補充帖子，來源：{source_id}，排序：{sort}")
                yield f"錯誤：在 {selected_source['source_name']} 中未找到相關帖子。請嘗試其他關鍵詞！"
                return
            filtered_supplemental = [
                item for item in supplemental_threads
                if any(kw.lower() in item["title"].lower() for kw in theme_keywords)
            ][:1]
            referenced_thread_ids = [str(item["thread_id"]) for item in filtered_supplemental]
            logger.debug(f"獲取補充帖子：{referenced_thread_ids}")
        
        prioritized_thread_data = {tid: thread_data_dict[tid] for tid in map(str, referenced_thread_ids) if tid in thread_data_dict}
        supplemental_thread_data = {tid: data for tid, data in thread_data_dict.items() if tid not in map(str, referenced_thread_ids)}
        thread_data_dict = {**prioritized_thread_data, **supplemental_thread_data}

    filtered_thread_data = {}
    total_replies_count = 0
    
    for tid, data in thread_data_dict.items():
        try:
            replies = data.get("replies", [])
            if not isinstance(replies, list):
                logger.warning(f"無效的回覆格式，帖子 ID={tid}：預期 list，得到 {type(replies)}")
                replies = []
            filtered_replies = []
            for r in replies:
                if not isinstance(r, dict) or not r.get("msg"):
                    continue
                cleaned_msg = clean_html(r["msg"])
                if len(cleaned_msg.strip()) <= 7 or cleaned_msg in ["[圖片]", "[無內容]", "[表情符號]"]:
                    continue
                r["reply_time"] = unix_to_readable(r.get("reply_time", "0"), context=f"reply in thread {tid}")
                filtered_replies.append(r)
            
            sorted_replies = sorted(
                filtered_replies,
                key=lambda x: x.get("like_count", 0),
                reverse=True
            )[:max_replies_per_thread]
            
            total_replies_count += len(sorted_replies)
            filtered_thread_data[tid] = {
                "thread_id": data.get("thread_id", tid),
                "title": data.get("title", ""),
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": unix_to_readable(data.get("last_reply_time", "0"), context=f"thread {tid}"),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": sorted_replies,
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(sorted_replies)
            }
        except Exception as e:
            logger.error(f"處理帖子 ID={tid} 失敗：{str(e)}")
            yield f"錯誤：處理帖子（ID={tid}）失敗（{str(e)}）。請聯繫支持。"
            return
    
    if total_replies_count < max_replies_per_thread and any(intent in ["follow_up", "fetch_thread_by_id", "analyze_sentiment"] for intent in intents):
        for tid, data in filtered_thread_data.items():
            if data["total_fetched_replies"] < max_replies_per_thread:
                async with request_semaphore:
                    if source_type == "lihkg":
                        content_result = await get_lihkg_thread_content(
                            thread_id=tid,
                            cat_id=source_id,
                            max_replies=max_replies_per_thread - data["total_fetched_replies"],
                            fetch_last_pages=1,
                            start_page=max(data["fetched_pages"], default=0) + 1
                        )
                    else:
                        content_result = await get_reddit_thread_content(
                            post_id=tid,
                            subreddit=source_id,
                            max_comments=max_comments
                        )
                if content_result.get("replies"):
                    total_replies = content_result.get("total_replies", data["no_of_reply"])
                    cleaned_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"additional reply in thread {tid}")
                        }
                        for reply in content_result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                    ]
                    filtered_additional_replies = [
                        r for r in cleaned_replies
                        if len(r["msg"].strip()) > 7
                    ]
                    updated_data = {
                        "thread_id": data.get("thread_id", tid),
                        "title": data.get("title", ""),
                        "no_of_reply": total_replies,
                        "last_reply_time": unix_to_readable(content_result.get("last_reply_time", data["last_reply_time"]), context=f"thread {tid}"),
                        "like_count": data.get("like_count", 0),
                        "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": data.get("replies", []) + filtered_additional_replies,
                        "fetched_pages": list(set(data.get("fetched_pages", []) + content_result.get("fetched_pages", []))),
                        "total_fetched_replies": len(data.get("replies", []) + filtered_additional_replies)
                    }
                    filtered_thread_data[tid] = updated_data
                    total_replies_count += len(filtered_additional_replies)
                    async with cache_lock:
                        st.session_state.thread_cache[tid] = {
                            "data": updated_data,
                            "timestamp": time.time()
                        }

    if not any(data["replies"] for data in filtered_thread_data.values()) and metadata:
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": unix_to_readable(data.get("last_reply_time", "0"), context=f"thread {tid}"),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": [],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": 0
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = 0
    
    primary_intent_info = max(intents_info, key=lambda x: x["confidence"]) if intents_info else {"intent": "summarize_posts", "confidence": 0.7}
    primary_intent = primary_intent_info["intent"]
    
    prompt = await build_dynamic_prompt(
        query=user_query,
        conversation_context=conversation_context,
        metadata=metadata,
        thread_data=list(filtered_thread_data.values()),
        filters=filters,
        intent=primary_intent,
        selected_source=selected_source,
        grok3_api_key=GROK3_API_KEY
    )
    
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4
    logger.info(
        f"生成提示：函數=stream_grok3_response, 查詢={user_query}, "
        f"提示長度={prompt_length} 字符, 估計 token={estimated_tokens}, "
        f"thread_data 帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}, "
        f"intents={intents}, 主要意圖={primary_intent}, source={selected_source}"
    )
    
    # Aggressive prompt size reduction
    reduction_attempts = 0
    while prompt_length > GROK3_TOKEN_LIMIT * 0.9 and reduction_attempts < 3:
        max_replies_per_thread = max_replies_per_thread // 2
        total_replies_count = 0
        filtered_thread_data = {
            tid: {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data.get("no_of_reply", 0),
                "last_reply_time": unix_to_readable(data.get("last_reply_time", "0"), context=f"thread {tid}"),
                "like_count": data.get("like_count", 0),
                "dislike_count": data.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "replies": data["replies"][:max_replies_per_thread],
                "fetched_pages": data.get("fetched_pages", []),
                "total_fetched_replies": len(data["replies"][:max_replies_per_thread])
            } for tid, data in filtered_thread_data.items()
        }
        total_replies_count = sum(len(data["replies"]) for data in filtered_thread_data.values())
        prompt = await build_dynamic_prompt(
            query=user_query,
            conversation_context=conversation_context,
            metadata=metadata,
            thread_data=list(filtered_thread_data.values()),
            filters=filters,
            intent=primary_intent,
            selected_source=selected_source,
            grok3_api_key=GROK3_API_KEY
        )
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4
        logger.info(
            f"縮減後提示（嘗試 {reduction_attempts + 1}）：函數=stream_grok3_response, 查詢={user_query}, "
            f"提示長度={prompt_length} 字符, 估計 token={estimated_tokens}, "
            f"thread_data 帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}, "
            f"intents={intents}, 主要意圖={primary_intent}, source={selected_source}"
        )
        reduction_attempts += 1

    if prompt_length > GROK3_TOKEN_LIMIT:
        logger.error(f"無法縮減提示至限制以下：最終長度={prompt_length} > {GROK3_TOKEN_LIMIT}")
        yield f"錯誤：提示數據過大，無法生成回應。請縮減查詢範圍或聯繫支持。"
        return

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    response_content = ""
    prompt_tokens = 0
    completion_tokens = 0
    
    messages = [
        {"role": "system", "content": (
            "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
            "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
            "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如列表、段落、表格等），"
            "確保結構清晰、內容連貫，且適合查詢的需求。"
        )},
        *conversation_context,
        {"role": "user", "content": prompt}
    ]
    payload = {
        "model": "grok-3",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True
    }
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                status_code = response.status
                if status_code != 200:
                    logger.error(f"回應生成失敗：狀態碼={status_code}, 意圖={intents}, source={selected_source}")
                    yield f"錯誤：生成回應失敗（狀態碼 {status_code}）。請稍後重試。"
                    return
                
                async for line in response.content:
                    if line and not line.isspace():
                        line_str = line.decode('utf-8').strip()
                        if line_str == "data: [DONE]":
                            logger.info(
                                f"Grok3 API 調用：函數=stream_grok3_response, "
                                f"查詢={user_query}, 意圖={intents}, 狀態碼={status_code}, "
                                f"輸入 token={prompt_tokens}, 輸出 token={completion_tokens}, "
                                f"回應長度={len(response_content)}, source={selected_source}"
                            )
                            break
                        if line_str.startswith("data:"):
                            try:
                                chunk = json.loads(line_str[6:])
                                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                if content:
                                    if "Content Moderation" in content or "Blocked" in content:
                                        logger.warning(f"檢測到內容審核：{content}")
                                        raise ValueError("檢測到內容審核")
                                    cleaned_content = clean_response(content)
                                    response_content += cleaned_content
                                    yield cleaned_content
                                usage = chunk.get("usage", {})
                                if usage:
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                            except json.JSONDecodeError:
                                logger.warning(f"流式數據 JSON 解碼錯誤：{line_str}")
                                continue
        except aiohttp.ClientConnectionError as e:
            logger.error(f"回應生成失敗：網絡連接錯誤，意圖={intents}, 錯誤={str(e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
            yield f"錯誤：網絡連接失敗（{str(e)}）。請檢查網絡或稍後重試。"
        except aiohttp.ClientResponseError as e:
            logger.error(f"回應生成失敗：API 響應錯誤，意圖={intents}, 錯誤={str(e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
            yield f"錯誤：API 響應錯誤（{str(e)}）。請稍後重試。"
        except asyncio.TimeoutError as e:
            logger.error(f"回應生成失敗：API 超時，意圖={intents}, 錯誤={str(e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
            # Fallback to non-streaming API call
            logger.info(f"嘗試非流式 API 調用作為回退，max_tokens={max_tokens // 2}")
            payload["stream"] = False
            payload["max_tokens"] = max_tokens // 2
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    status_code = response.status
                    if status_code != 200:
                        logger.error(f"非流式回退失敗：狀態碼={status_code}, 意圖={intents}, source={selected_source}")
                        yield f"錯誤：生成回應失敗（狀態碼 {status_code}）。請稍後重試。"
                        return
                    data = await response.json()
                    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        cleaned_content = clean_response(content)
                        response_content += cleaned_content
                        yield cleaned_content
                    usage = data.get("usage", {})
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                    logger.info(
                        f"非流式回退成功：函數=stream_grok3_response, "
                        f"查詢={user_query}, 意圖={intents}, 狀態碼={status_code}, "
                        f"輸入 token={prompt_tokens}, 輸出 token={completion_tokens}, "
                        f"回應長度={len(response_content)}, source={selected_source}"
                    )
            except Exception as fallback_e:
                logger.error(f"非流式回退失敗：意圖={intents}, 錯誤={str(fallback_e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
                yield f"錯誤：生成回應失敗（{str(fallback_e)}）。請稍後重試或聯繫支持。"
        except Exception as e:
            logger.error(f"回應生成失敗：意圖={intents}, 錯誤={str(e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
            yield f"錯誤：生成回應失敗（{str(e)}）。請稍後重試或聯繫支持。"
        finally:
            await session.close()

def clean_cache(max_age=3600):
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]
    if len(st.session_state.thread_cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(
            st.session_state.thread_cache.items(),
            key=lambda x: x[1]["timestamp"]
        )
        for key, _ in sorted_keys[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE]:
            del st.session_state.thread_cache[key]
        logger.info(f"清理緩存，移除 {len(sorted_keys[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE])} 個過舊條目，當前緩存大小：{len(st.session_state.thread_cache)}")

def unix_to_readable(timestamp, context="unknown"):
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        elif isinstance(timestamp, str):
            try:
                timestamp_int = int(timestamp)
                dt = datetime.datetime.fromtimestamp(timestamp_int, tz=HONG_KONG_TZ)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                try:
                    dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                    dt = HONG_KONG_TZ.localize(dt)
                    return dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    raise ValueError(f"無法解析日期字符串：{timestamp}")
        else:
            raise TypeError(f"無效的時間戳類型：{type(timestamp)}")
    except (ValueError, TypeError) as e:
        logger.warning(f"無法轉換時間戳：值={timestamp}, 類型={type(timestamp)}, 上下文={context}, 錯誤={str(e)}")
        return "1970-01-01 00:00:00"

def configure_lihkg_api_logger():
    configure_logger("lihkg_api", "lihkg_api.log")

def configure_reddit_api_logger():
    configure_logger("reddit_api", "reddit_api.log")

async def process_user_question(user_query, selected_source, source_id, source_type="lihkg", analysis=None, request_counter=0, last_reset=0, rate_limit_until=0, conversation_context=None, progress_callback=None):
    if source_type == "lihkg":
        configure_lihkg_api_logger()
    else:
        configure_reddit_api_logger()
    
    if isinstance(selected_source, str):
        selected_source = {
            "source_name": selected_source,
            "source_type": source_type
        }
        logger.debug(f"將字符串 selected_source 轉換為字典：{selected_source}")
    elif not isinstance(selected_source, dict):
        logger.warning(f"無效的 selected_source 類型：{type(selected_source)}，使用默認值")
        selected_source = {"source_name": "未知", "source_type": source_type}
    elif "source_name" not in selected_source or "source_type" not in selected_source:
        logger.warning(f"selected_source 缺少必要字段：{selected_source}，補充默認值")
        selected_source = {
            "source_name": selected_source.get("source_name", "未知"),
            "source_type": selected_source.get("source_type", source_type)
        }
    
    try:
        clean_cache()
        
        if rate_limit_until > time.time():
            logger.warning(f"速率限制生效，直到 {datetime.datetime.fromtimestamp(rate_limit_until, tz=HONG_KONG_TZ):%Y-%m-%d %H:%M:%S}")
            return {
                "selected_source": selected_source,
                "thread_data": [],
                "rate_limit_info": [{"message": "速率限制生效", "until": rate_limit_until}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        try:
            GROK3_API_KEY = st.secrets["grok3key"]
        except KeyError as e:
            logger.error(f"缺少 Grok 3 API 密鑰：{str(e)}")
            return {
                "selected_source": selected_source,
                "thread_data": [],
                "rate_limit_info": [{"message": "缺少 API 密鑰"}],
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        if not analysis:
            analysis = await analyze_and_screen(user_query, selected_source["source_name"], source_id, source_type, conversation_context)
        
        post_limit = min(analysis.get("post_limit", 5), 20)
        filters = analysis.get("filters", {})
        min_replies = filters.get("min_replies", 10)
        top_thread_ids = list(set(analysis.get("top_thread_ids", [])))
        intents = analysis.get("intents", [{"intent": "summarize_posts", "confidence": 0.7}])
        
        logger.info(f"處理用戶問題：查詢={user_query}, intents={[i['intent'] for i in intents]}, source_type={source_type}, source_id={source_id}, post_limit={post_limit}, top_thread_ids={top_thread_ids}")
        
        keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
        fetch_last_pages = 1 if keyword_result.get("time_sensitive", False) else 0
        sort = "new" if keyword_result.get("time_sensitive", False) else "best"
        logger.debug(f"選擇排序方式：{sort}，基於 time_sensitive={keyword_result.get('time_sensitive', False)}")
        
        max_comments = 300 if source_type == "reddit" and any(i["intent"] == "follow_up" for i in intents) else 100
        max_replies = 300 if any(i["intent"] in ["follow_up", "analyze_sentiment"] for i in intents) else 100
        
        if any(i["intent"] in ["fetch_thread_by_id", "follow_up", "analyze_sentiment"] for i in intents) and top_thread_ids:
            thread_data = []
            rate_limit_info = []
            processed_thread_ids = set()
            
            candidate_threads = [{"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0} for tid in top_thread_ids]
            
            tasks = []
            for idx, thread_id in enumerate(top_thread_ids):
                thread_id_str = str(thread_id)
                if thread_id_str in processed_thread_ids:
                    logger.debug(f"跳過重複的 thread_id={thread_id_str}")
                    continue
                processed_thread_ids.add(thread_id_str)
                async with cache_lock:
                    if thread_id_str in st.session_state.thread_cache and st.session_state.thread_cache[thread_id_str]["data"].get("replies"):
                        cached_data = st.session_state.thread_cache[thread_id_str]["data"]
                        thread_data.append(cached_data)
                        logger.debug(f"緩存命中：thread_id={thread_id_str}")
                        continue
                if source_type == "lihkg":
                    tasks.append(get_lihkg_thread_content(
                        thread_id=thread_id_str,
                        cat_id=source_id,
                        max_replies=max_replies,
                        fetch_last_pages=fetch_last_pages,
                        specific_pages=[],
                        start_page=1
                    ))
                else:
                    tasks.append(get_reddit_thread_content(
                        post_id=thread_id_str,
                        subreddit=source_id,
                        max_comments=max_comments
                    ))
            
            if tasks:
                content_results = await asyncio.gather(*tasks, return_exceptions=True)
                for idx, result in enumerate(content_results):
                    if isinstance(result, Exception):
                        logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']}：{str(result)}")
                        continue
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    
                    thread_id = str(candidate_threads[idx]["thread_id"])
                    if result.get("title"):
                        total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                        if total_replies == 0:
                            total_replies = candidate_threads[idx]["no_of_reply"]
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"reply in thread {thread_id}")
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                            and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                            "like_count": result.get("like_count", 0),
                            "dislike_count": result.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies)
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id] = {
                                "data": thread_info,
                                "timestamp": time.time()
                            }
            
            if len(thread_data) < 5 and any(i["intent"] == "follow_up" for i in intents):
                keyword_result = await extract_keywords(user_query, conversation_context, GROK3_API_KEY)
                theme_keywords = keyword_result["keywords"]
                
                async with request_semaphore:
                    if source_type == "lihkg":
                        supplemental_result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=1,
                            max_pages=2
                        )
                    else:
                        supplemental_result = await get_reddit_topic_list(
                            subreddit=source_id,
                            start_page=1,
                            max_pages=2,
                            sort=sort
                        )
                supplemental_threads = supplemental_result.get("items", [])
                if not supplemental_threads:
                    logger.warning(f"無法獲取補充帖子，來源：{source_id}，排序：{sort}")
                    return {
                        "selected_source": selected_source,
                        "thread_data": thread_data,
                        "rate_limit_info": rate_limit_info + [{"message": f"無法獲取補充帖子，來源：{source_id}"}],
                        "request_counter": request_counter,
                        "last_reset": last_reset,
                        "rate_limit_until": rate_limit_until,
                        "analysis": analysis
                    }
                filtered_supplemental = [
                    item for item in supplemental_threads
                    if str(item["thread_id"]) not in top_thread_ids
                    and any(kw.lower() in item["title"].lower() for kw in theme_keywords)
                ][:5 - len(thread_data)]
                request_counter = supplemental_result.get("request_counter", request_counter)
                last_reset = supplemental_result.get("last_reset", last_reset)
                rate_limit_until = supplemental_result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(supplemental_result.get("rate_limit_info", []))
                
                supplemental_tasks = []
                for item in filtered_supplemental:
                    thread_id = str(item["thread_id"])
                    if thread_id in processed_thread_ids:
                        logger.debug(f"跳過重複的補充 thread_id={thread_id}")
                        continue
                    processed_thread_ids.add(thread_id)
                    if source_type == "lihkg":
                        supplemental_tasks.append(get_lihkg_thread_content(
                            thread_id=thread_id,
                            cat_id=source_id,
                            max_replies=max_replies,
                            fetch_last_pages=fetch_last_pages,
                            specific_pages=[],
                            start_page=1
                        ))
                    else:
                        supplemental_tasks.append(get_reddit_thread_content(
                            post_id=thread_id,
                            subreddit=source_id,
                            max_comments=max_comments
                        ))
                
                if supplemental_tasks:
                    supplemental_results = await asyncio.gather(*supplemental_tasks, return_exceptions=True)
                    for idx, result in enumerate(supplemental_results):
                        if isinstance(result, Exception):
                            logger.warning(f"無法抓取補充帖子 {filtered_supplemental[idx]['thread_id']}：{str(result)}")
                            continue
                        request_counter = result.get("request_counter", request_counter)
                        last_reset = result.get("last_reset", last_reset)
                        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                        rate_limit_info.extend(result.get("rate_limit_info", []))
                        
                        thread_id = str(filtered_supplemental[idx]["thread_id"])
                        if result.get("title"):
                            total_replies = result.get("total_replies", filtered_supplemental[idx].get("no_of_reply", 0))
                            if total_replies == 0:
                                total_replies = filtered_supplemental[idx].get("no_of_reply", 0)
                            filtered_replies = [
                                {
                                    "reply_id": reply.get("reply_id"),
                                    "msg": clean_html(reply.get("msg", "[無內容]")),
                                    "like_count": reply.get("like_count", 0),
                                    "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                    "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"supplemental reply in thread {thread_id}")
                                }
                                for reply in result.get("replies", [])
                                if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                                and len(clean_html(reply.get("msg")).strip()) > 7
                            ]
                            thread_info = {
                                "thread_id": thread_id,
                                "title": result.get("title"),
                                "no_of_reply": total_replies,
                                "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                                "like_count": filtered_supplemental[idx].get("like_count", 0),
                                "dislike_count": filtered_supplemental[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "replies": filtered_replies,
                                "fetched_pages": result.get("fetched_pages", []),
                                "total_fetched_replies": len(filtered_replies)
                            }
                            thread_data.append(thread_info)
                            async with cache_lock:
                                st.session_state.thread_cache[thread_id] = {
                                    "data": thread_info,
                                    "timestamp": time.time()
                                }
            
            logger.info(
                f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}"
            )
            
            return {
                "selected_source": selected_source,
                "thread_data": thread_data,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": rate_limit_until,
                "analysis": analysis
            }
        
        thread_data = []
        rate_limit_info = []
        candidate_threads = []
        processed_thread_ids = []
        
        if top_thread_ids:
            candidate_threads = [
                {"thread_id": str(tid), "title": "", "no_of_reply": 0, "like_count": 0}
                for tid in top_thread_ids
            ]
        else:
            initial_threads = []
            for page in range(1, 4):
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_topic_list(
                            cat_id=source_id,
                            start_page=page,
                            max_pages=1
                        )
                    else:
                        result = await get_reddit_topic_list(
                            subreddit=source_id,
                            start_page=page,
                            max_pages=1,
                            sort=sort
                        )
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                items = result.get("items", [])
                initial_threads.extend(items)
                if not items:
                    logger.warning(f"未抓取到分類 ID={source_id}，頁面={page} 的帖子，排序={sort}")
                if len(initial_threads) >= 150:
                    initial_threads = initial_threads[:150]
                    break
                if progress_callback:
                    progress_callback(f"已抓取第 {page}/3 頁帖子", 0.1 + 0.2 * (page / 3))
            
            filtered_items = [
                item for item in initial_threads
                if item.get("no_of_reply", 0) >= min_replies
            ]
            
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                async with cache_lock:
                    if thread_id not in st.session_state.thread_cache:
                        cache_data = {
                            "thread_id": thread_id,
                            "title": item["title"],
                            "no_of_reply": item.get("no_of_reply", 0),
                            "last_reply_time": unix_to_readable(item.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": [],
                            "fetched_pages": []
                        }
                        st.session_state.thread_cache[thread_id] = {
                            "data": cache_data,
                            "timestamp": time.time()
                        }
            
            if any(i["intent"] == "fetch_dates" for i in intents):
                sorted_items = sorted(
                    filtered_items,
                    key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"),
                    reverse=True
                )
                candidate_threads = sorted_items[:post_limit]
            else:
                if filtered_items:
                    prioritization = await prioritize_threads_with_grok(
                        user_query, filtered_items, selected_source["source_name"], source_id, source_type, [i["intent"] for i in intents]
                    )
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    if not top_thread_ids:
                        sorted_items = sorted(
                            filtered_items,
                            key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4,
                            reverse=True
                        )
                        candidate_threads = sorted_items[:post_limit]
                    else:
                        candidate_threads = [
                            item for item in filtered_items
                            if str(item["thread_id"]) in map(str, top_thread_ids)
                        ][:post_limit]
        
        if progress_callback:
            progress_callback("正在抓取帖子內容", 0.3)
        
        tasks = []
        for idx, item in enumerate(candidate_threads):
            thread_id = str(item["thread_id"])
            if thread_id in processed_thread_ids:
                logger.debug(f"跳過重複的 thread_id={thread_id}")
                continue
            processed_thread_ids.append(thread_id)
            async with cache_lock:
                if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
                    cached_data = st.session_state.thread_cache[thread_id]["data"]
                    thread_data.append(cached_data)
                    logger.debug(f"緩存命中：thread_id={thread_id}")
                    continue
            if source_type == "lihkg":
                tasks.append((idx, get_lihkg_thread_content(
                    thread_id=thread_id,
                    cat_id=source_id,
                    max_replies=max_replies,
                    fetch_last_pages=fetch_last_pages,
                    specific_pages=[],
                    start_page=1
                )))
            else:
                tasks.append((idx, get_reddit_thread_content(
                    post_id=thread_id,
                    subreddit=source_id,
                    max_comments=max_comments
                )))
        
        if tasks:
            content_results = await asyncio.gather(*[task[1] for task in tasks], return_exceptions=True)
            for task_idx, result in enumerate(content_results):
                idx = tasks[task_idx][0]
                if isinstance(result, Exception):
                    logger.warning(f"無法抓取帖子 {candidate_threads[idx]['thread_id']} 內容：{str(result)}")
                    continue
                request_counter = result.get("request_counter", request_counter)
                last_reset = result.get("last_reset", last_reset)
                rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                rate_limit_info.extend(result.get("rate_limit_info", []))
                
                thread_id = str(candidate_threads[idx]["thread_id"])
                if result.get("title"):
                    total_replies = result.get("total_replies", candidate_threads[idx]["no_of_reply"])
                    if total_replies == 0:
                        total_replies = candidate_threads[idx]["no_of_reply"]
                    filtered_replies = [
                        {
                            "reply_id": reply.get("reply_id"),
                            "msg": clean_html(reply.get("msg", "[無內容]")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"reply in thread {thread_id}")
                        }
                        for reply in result.get("replies", [])
                        if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"]
                        and len(clean_html(reply.get("msg")).strip()) > 7
                    ]
                    thread_info = {
                        "thread_id": thread_id,
                        "title": result.get("title"),
                        "no_of_reply": total_replies,
                        "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                        "like_count": candidate_threads[idx].get("like_count", 0),
                        "dislike_count": candidate_threads[idx].get("dislike_count", 0) if source_type == "lihkg" else 0,
                        "replies": filtered_replies,
                        "fetched_pages": result.get("fetched_pages", []),
                        "total_fetched_replies": len(filtered_replies)
                    }
                    thread_data.append(thread_info)
                    async with cache_lock:
                        st.session_state.thread_cache[thread_id] = {
                            "data": thread_info,
                            "timestamp": time.time()
                        }
        
        if progress_callback:
            progress_callback("正在準備數據", 0.5)
        
        logger.info(
            f"最終 thread_data：{[{'thread_id': data['thread_id'], 'replies_count': len(data['replies'])} for data in thread_data]}, source={selected_source}"
        )
        
        return {
            "selected_source": selected_source,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
    
    except Exception as e:
        logger.error(f"處理用戶問題失敗：查詢={user_query}, 錯誤={str(e)}, source={selected_source}, 堆棧={traceback.format_exc()}")
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": [{"message": f"處理錯誤：{str(e)}"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis
        }
