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
from dynamic_prompt_utils import build_dynamic_prompt, parse_query, extract_keywords, CONFIG, INTENT_CONFIG

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "grok_processing.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 270000
API_TIMEOUT = 120
MAX_CACHE_SIZE = 100

cache_lock = asyncio.Lock()
request_semaphore = asyncio.Semaphore(5)

def clean_html(text):
    if not isinstance(text, str):
        text = str(text)
    try:
        clean = re.compile(r"<[^>]+>")
        text = clean.sub("", text)
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "[表情符號]" if "hkgmoji" in text else "[圖片]" if any(ext in text.lower() for ext in [".webp", ".jpg", ".png"]) else "[無內容]"
        return text
    except Exception as e:
        logger.error(f"HTML 清理失敗：{str(e)}")
        return text

def clean_response(response):
    if isinstance(response, str):
        cleaned = re.sub(r"\[post_id: [a-f0-9]{40}\]", "[回覆]", response)
        return cleaned
    return response

def unix_to_readable(timestamp, context="unknown"):
    try:
        if isinstance(timestamp, (int, float)):
            dt = datetime.datetime.fromtimestamp(timestamp, tz=HONG_KONG_TZ)
        elif isinstance(timestamp, str):
            try:
                timestamp_int = int(timestamp)
                dt = datetime.datetime.fromtimestamp(timestamp_int, tz=HONG_KONG_TZ)
            except ValueError:
                dt = datetime.datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
                dt = HONG_KONG_TZ.localize(dt)
        else:
            raise TypeError(f"無效的時間戳類型：{type(timestamp)}")
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError) as e:
        logger.warning(f"無法轉換時間戳：值={timestamp}, 上下文={context}, 錯誤={str(e)}")
        return "1970-01-01 00:00:00"

def normalize_selected_source(selected_source, source_type):
    if isinstance(selected_source, str):
        return {"source_name": selected_source, "source_type": source_type}
    if not isinstance(selected_source, dict) or "source_name" not in selected_source or "source_type" not in selected_source:
        logger.warning(f"無效的 selected_source：{selected_source}，使用默認值")
        return {"source_name": "未知", "source_type": source_type}
    return selected_source

async def summarize_context(conversation_context):
    if not conversation_context:
        return {"theme": "一般", "keywords": []}
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("缺少 Grok 3 API 密鑰")
        return {"theme": "一般", "keywords": []}
    prompt = f"""
你是對話摘要助手，請分析以下對話歷史，提煉主要主題和關鍵詞（最多3個）。
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
輸出格式：{{"theme": "主要主題", "keywords": ["關鍵詞1", "關鍵詞2", "關鍵詞3"]}}
"""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 100, "temperature": 0.5}
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                if response.status != 200:
                    logger.warning(f"對話摘要失敗：狀態碼={response.status}")
                    return {"theme": "一般", "keywords": []}
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"對話摘要成功：result={result}")
                return result
        except Exception as e:
            logger.warning(f"對話摘要錯誤：{str(e)}")
            return {"theme": "一般", "keywords": []}

async def analyze_and_screen(user_query, source_name, source_id, source_type="lihkg", conversation_context=None):
    conversation_context = conversation_context or []
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("缺少 Grok 3 API 密鑰")
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
            "theme_keywords": [],
        }
    logger.info(f"開始語義分析：查詢={user_query}")
    parsed_query = await parse_query(user_query, conversation_context, api_key, source_type)
    intents = parsed_query.get("intents", [])
    if not intents:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "無法識別有效意圖，默認總結"}]
        logger.warning(f"無有效意圖：查詢={user_query}，回退到 summarize_posts")
    query_keywords = parsed_query.get("keywords", [])
    top_thread_ids = parsed_query.get("thread_ids", [])
    reason = parsed_query.get("reason", "無原因")
    confidence = parsed_query.get("confidence", 0.5)
    post_limit = max(3, min(15, parsed_query.get("post_limit", 5)))  # 確保 post_limit 在 3 到 15
    context_summary = await summarize_context(conversation_context)
    historical_theme = context_summary.get("theme", "一般")
    historical_keywords = context_summary.get("keywords", [])
    is_vague = len(query_keywords) < 2 and not any(keyword in user_query for keyword in ["分析", "總結", "討論", "主題", "時事"]) and not any(i["intent"] == "list_titles" and i["confidence"] >= 0.9 for i in intents)
    if is_vague:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": f"問題模糊，歷史主題：{historical_theme}" if historical_theme != "一般" else "問題模糊，默認總結"}]
        reason = intents[0]["reason"]
    theme = historical_theme if is_vague else (query_keywords[0] if query_keywords else "一般")
    theme_keywords = historical_keywords if is_vague else query_keywords
    primary_intent = max(intents, key=lambda x: x["confidence"])["intent"]
    # 內聯 get_intent_processing_params 邏輯
    intent_params = INTENT_CONFIG.get(primary_intent, INTENT_CONFIG["summarize_posts"]).get("processing", {}).copy()
    sort_override = intent_params.get("sort_override", {})
    if source_type.lower() in sort_override:
        intent_params["sort"] = sort_override[source_type.lower()]
    if source_type.lower() == "lihkg" and intent_params.get("sort") == "confidence":
        intent_params["sort"] = "hot"
    logger.info(f"動態 post_limit 來自 parse_query: {post_limit}, 意圖: {primary_intent}")
    return {
        "direct_response": primary_intent in ["general_query", "introduce"],
        "intents": intents,
        "theme": theme,
        "source_type": source_type,
        "source_ids": [source_id],
        "data_type": intent_params.get("data_type", "both"),
        "post_limit": post_limit,
        "filters": {
            "min_replies": intent_params.get("min_replies", 10),
            "min_likes": 0,
            "sort": intent_params.get("sort", "hot"),
            "keywords": theme_keywords,
        },
        "processing": {"intents": [i["intent"] for i in intents], "top_thread_ids": top_thread_ids, "analysis": parsed_query},
        "candidate_thread_ids": top_thread_ids,
        "top_thread_ids": top_thread_ids,
        "needs_advanced_analysis": confidence < 0.7,
        "reason": reason,
        "theme_keywords": theme_keywords,
    }

async def prioritize_threads_with_grok(user_query, threads, source_name, source_id, source_type="lihkg", intents=["summarize_posts"], post_limit=5):
    logger.info(f"正在排序帖子：查詢={user_query}, 帖子數={len(threads)}, 意圖={intents}, post_limit={post_limit}")
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("缺少 Grok 3 API 密鑰")
        return {"top_thread_ids": [], "reason": "缺少 API 密鑰", "intent_breakdown": []}
    
    max_threads = 50
    threads = threads[:max_threads]
    logger.info(f"限制帖子數到 {len(threads)} 以避免提示大小問題")
    
    if any(intent == "follow_up" for intent in intents):
        referenced_thread_ids = re.findall(r"\[帖子 ID: (\w+)\]", st.session_state.get("conversation_context", [])[-1].get("content", "") if st.session_state.get("conversation_context") else "")
        valid_ids = [tid for tid in referenced_thread_ids if any(str(t["thread_id"]) == tid for t in threads)]
        if valid_ids:
            valid_ids = valid_ids[:post_limit]  # 尊重 post_limit
            logger.info(f"檢測到追問意圖，使用參考帖子 ID: {valid_ids}")
            return {"top_thread_ids": valid_ids, "reason": "追問參考帖子", "intent_breakdown": [{"intent": "follow_up", "thread_ids": valid_ids}]}
    
    threads = [{"thread_id": str(t["thread_id"]), **t} for t in threads]
    prompt = f"""
你是帖子優先級排序助手，請根據用戶查詢和意圖，從提供的帖子中選出最多20個最相關的帖子。
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
"""
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4
    logger.info(f"創建提示：長度={prompt_length} 字符, 估計 token={estimated_tokens}, 提示預覽={prompt[:200]}...")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {"model": "grok-3", "messages": [{"role": "user", "content": prompt}], "max_tokens": 500, "temperature": 0.7}
    logger.info(f"API payload: {json.dumps(payload, ensure_ascii=False)[:200]}...")
    
    async with aiohttp.ClientSession() as session:
        for attempt in range(2):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.warning(f"API 調用失敗：狀態碼={response.status}, 嘗試={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"API 調用失敗：無選擇，嘗試={attempt + 1}")
                        continue
                    response_content = data["choices"][0]["message"]["content"]
                    logger.info(f"原始 API 回應：{response_content[:500]}...")
                    try:
                        result = json.loads(response_content)
                        top_thread_ids = [str(tid) for tid in result.get("top_thread_ids", []) if str(tid) in [str(t["thread_id"]) for t in threads]][:20]  # 最多 20 個
                        top_thread_ids = top_thread_ids[:post_limit]  # 根據動態 post_limit 裁剪
                        logger.info(f"API 返回 top_thread_ids: {top_thread_ids}, 原因={result.get('reason', '無原因')}, 最終選取數量={len(top_thread_ids)}")
                        return {
                            "top_thread_ids": top_thread_ids,
                            "reason": result.get("reason", "無原因"),
                            "intent_breakdown": result.get("intent_breakdown", []),
                        }
                    except json.JSONDecodeError as e:
                        logger.error(f"JSON 解碼錯誤：{str(e)}, response_content={response_content[:200]}...")
                        continue
            except Exception as e:
                logger.error(f"API 調用錯誤：{str(e)}, 嘗試={attempt + 1}")
                if attempt < 1:
                    await asyncio.sleep(2)
                    continue
                sorted_threads = sorted(threads, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)
                top_thread_ids = [str(t["thread_id"]) for t in sorted_threads[:20]][:post_limit]  # 回退時也尊重 post_limit
                logger.info(f"回退到熱門度排序：top_thread_ids={top_thread_ids}, 原因=API 失敗 ({str(e)}), 最終選取數量={len(top_thread_ids)}")
                return {
                    "top_thread_ids": top_thread_ids,
                    "reason": f"排序失敗（{str(e)}），回退到熱門度排序",
                    "intent_breakdown": [],
                }
    logger.warning(f"排序失敗，所有嘗試均失敗")
    return {"top_thread_ids": [], "reason": "排序失敗，所有嘗試均失敗", "intent_breakdown": []}

async def stream_grok3_response(user_query, metadata, thread_data, processing, selected_source, conversation_context=None, needs_advanced_analysis=False, reason="", filters=None, source_id=None, source_type="lihkg"):
    conversation_context = conversation_context or []
    filters = filters or {"min_replies": 10, "min_likes": 0}
    selected_source = normalize_selected_source(selected_source, source_type)
    if not thread_data:
        error_msg = (
            f"在 {selected_source['source_name']} 中未找到符合條件的帖子。\n"
            f"查詢：{user_query}\n"
            f"篩選條件：{json.dumps(filters, ensure_ascii=False)}\n"
            f"建議：嘗試不同的關鍵詞或更廣泛的查詢範圍。"
        )
        logger.warning(error_msg)
        yield error_msg
        return
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        logger.error("缺少 Grok 3 API 密鑰")
        yield "錯誤：缺少 API 密鑰"
        return
    intents_info = processing.get("analysis", {}).get("intents", [{"intent": "summarize_posts", "confidence": 0.7, "reason": "默認意圖"}])
    intents = [i["intent"] for i in intents_info if i["intent"] is not None]
    if not intents:
        intents = ["summarize_posts"]
        logger.warning(f"無有效意圖，回退到：{intents}")
    primary_intent = max(intents_info, key=lambda x: x["confidence"])["intent"]
    # 內聯 get_intent_processing_params 邏輯
    intent_params = INTENT_CONFIG.get(primary_intent, INTENT_CONFIG["summarize_posts"]).get("processing", {}).copy()
    sort_override = intent_params.get("sort_override", {})
    if source_type.lower() in sort_override:
        intent_params["sort"] = sort_override[source_type.lower()]
    if source_type.lower() == "lihkg" and intent_params.get("sort") == "confidence":
        intent_params["sort"] = "hot"
    logger.info(f"開始生成回應：查詢={user_query}, 意圖={intents}, 來源={selected_source}")
    total_min_tokens = sum(INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])["word_range"][0] / 0.8 for intent in intents)
    total_max_tokens = sum(INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])["word_range"][1] / 0.8 for intent in intents)
    prompt_length = len(json.dumps(thread_data, ensure_ascii=False)) + len(user_query) + 1000
    length_factor = min(prompt_length / GROK3_TOKEN_LIMIT, 1.0)
    max_tokens = min(int(total_min_tokens + (total_max_tokens - total_min_tokens) * length_factor) + 1000, 25000)
    max_replies_per_thread = intent_params.get("max_replies", 150)
    thread_data_dict = {str(data["thread_id"]): data for data in thread_data} if isinstance(thread_data, list) else thread_data
    filtered_thread_data = {}
    total_replies_count = 0
    for tid, data in thread_data_dict.items():
        replies = data.get("replies", [])
        filtered_replies = [
            {
                "reply_id": r.get("reply_id"),
                "msg": clean_html(r.get("msg", "[無內容]")),
                "like_count": r.get("like_count", 0),
                "dislike_count": r.get("dislike_count", 0) if source_type == "lihkg" else 0,
                "reply_time": unix_to_readable(r.get("reply_time", "0"), context=f"reply in thread {tid}"),
            }
            for r in replies
            if r.get("msg") and clean_html(r.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"] and len(clean_html(r.get("msg")).strip()) > 7
        ]
        sorted_replies = sorted(filtered_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:max_replies_per_thread]
        total_replies_count += len(sorted_replies)
        filtered_thread_data[tid] = {
            **data,
            "replies": sorted_replies,
            "total_fetched_replies": len(sorted_replies),
            "last_reply_time": unix_to_readable(data.get("last_reply_time", "0"), context=f"thread {tid}"),
            "no_of_reply": data.get("no_of_reply", 0),
        }
    prompt = await build_dynamic_prompt(user_query, conversation_context, metadata, list(filtered_thread_data.values()), filters, primary_intent, selected_source, api_key)
    prompt_length = len(prompt)
    estimated_tokens = prompt_length // 4
    prompt_summary = prompt[:100] + "..." if prompt_length > 100 else prompt
    if "prompt_stats" not in st.session_state:
        st.session_state.prompt_stats = {"lengths": [], "max_length": 0, "min_length": float("inf"), "count": 0}
    st.session_state.prompt_stats["lengths"].append(prompt_length)
    st.session_state.prompt_stats["max_length"] = max(st.session_state.prompt_stats["max_length"], prompt_length)
    st.session_state.prompt_stats["min_length"] = min(st.session_state.prompt_stats["min_length"], prompt_length)
    st.session_state.prompt_stats["count"] += 1
    avg_length = sum(st.session_state.prompt_stats["lengths"]) / st.session_state.prompt_stats["count"] if st.session_state.prompt_stats["count"] > 0 else 0
    logger.info(
        f"提示長度統計：當前={prompt_length}, 平均={avg_length:.2f}, 最大={st.session_state.prompt_stats['max_length']}, "
        f"最小={st.session_state.prompt_stats['min_length']}, 總次數={st.session_state.prompt_stats['count']}"
    )
    reduction_attempts = 0
    while prompt_length > GROK3_TOKEN_LIMIT * 0.95 and reduction_attempts < 2:
        max_replies_per_thread = max_replies_per_thread // 2 or 10
        total_replies_count = 0
        for tid, data in filtered_thread_data.items():
            replies = data.get("replies", [])[:max_replies_per_thread]
            total_replies_count += len(replies)
            filtered_thread_data[tid]["replies"] = replies
            filtered_thread_data[tid]["total_fetched_replies"] = len(replies)
        prompt = await build_dynamic_prompt(user_query, conversation_context, metadata, list(filtered_thread_data.values()), filters, primary_intent, selected_source, api_key)
        prompt_length = len(prompt)
        estimated_tokens = prompt_length // 4
        prompt_summary = prompt[:100] + "..." if prompt_length > 100 else prompt
        logger.info(f"提示縮減：嘗試={reduction_attempts + 1}, 新長度={prompt_length}, 保留帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}")
        reduction_attempts += 1
    if prompt_length > GROK3_TOKEN_LIMIT:
        logger.error(f"提示長度 {prompt_length} 超過限制 {GROK3_TOKEN_LIMIT}，無法進一步縮減")
        yield "錯誤：提示過大，請縮減查詢範圍或聯繫 xAI 支持：https://x.ai/api。"
        return
    logger.info(f"生成提示：查詢={user_query}, 提示長度={prompt_length} 字符, 估計 token={estimated_tokens}, 帖子數={len(filtered_thread_data)}, 總回覆數={total_replies_count}, 意圖={intents}, 提示摘要={prompt_summary}")
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "grok-3",
        "messages": [
            {
                "role": "system",
                "content": f"你是社交媒體數據助手，以繁體中文回答，語氣客觀輕鬆，使用 [帖子 ID: {{thread_id}}] 格式引用帖子。",
            },
            *conversation_context,
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }
    async with aiohttp.ClientSession() as session:
        for attempt in range(2):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                    if response.status != 200:
                        logger.error(f"API 失敗：狀態碼={response.status}, 嘗試={attempt + 1}")
                        if attempt < 1:
                            await asyncio.sleep(2)
                        continue
                    response_content = ""
                    prompt_tokens = 0
                    completion_tokens = 0
                    async for line in response.content:
                        if line and not line.isspace():
                            line_str = line.decode("utf-8").strip()
                            if line_str == "data: [DONE]":
                                break
                            if line_str.startswith("data:"):
                                try:
                                    chunk = json.loads(line_str[6:])
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content:
                                        cleaned_content = clean_response(content)
                                        response_content += cleaned_content
                                        yield cleaned_content
                                    usage = chunk.get("usage", {})
                                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                                    completion_tokens = usage.get("completion_tokens", completion_tokens)
                                except json.JSONDecodeError:
                                    continue
                    logger.info(f"流式 API 成功：回應長度={len(response_content)}, 提示 token={prompt_tokens}, 完成 token={completion_tokens}")
                    return
            except (aiohttp.ClientConnectionError, aiohttp.ClientResponseError, asyncio.TimeoutError) as e:
                logger.error(f"流式 API 錯誤：{str(e)}, 嘗試={attempt + 1}")
                if attempt < 1:
                    await asyncio.sleep(2)
                    continue
                logger.info(f"回退到非流式 API，max_tokens={max_tokens // 2}")
                payload["stream"] = False
                payload["max_tokens"] = max_tokens // 2
                try:
                    async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=API_TIMEOUT) as response:
                        if response.status != 200:
                            logger.error(f"非流式 API 失敗：狀態碼={response.status}")
                            yield f"錯誤：生成回應失敗（狀態碼 {response.status}）。請稍後重試。"
                            return
                        data = await response.json()
                        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        cleaned_content = clean_response(content)
                        logger.info(f"非流式 API 成功：回應長度={len(cleaned_content)}, 提示 token={data.get('usage', {}).get('prompt_tokens', 0)}")
                        yield cleaned_content
                        return
                except Exception as e:
                    logger.error(f"非流式 API 錯誤：{str(e)}")
                    yield f"錯誤：生成回應失敗（{str(e)}）。請檢查網絡或聯繫 xAI 支持：https://x.ai/api。"
                    return
        yield f"錯誤：生成回應失敗（多次嘗試失敗）。請檢查網絡或聯繫 xAI 支持：https://x.ai/api。"

async def process_user_question(user_query, selected_source, source_id, source_type="lihkg", analysis=None, request_counter=0, last_reset=0, rate_limit_until=0, conversation_context=None, progress_callback=None):
    if source_type == "lihkg":
        configure_lihkg_api_logger()
    else:
        configure_reddit_api_logger()
    selected_source = normalize_selected_source(selected_source, source_type)
    clean_cache()
    if rate_limit_until > time.time():
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": [{"message": "速率限制生效", "until": rate_limit_until}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis,
        }
    try:
        api_key = st.secrets["grok3key"]
    except KeyError:
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": [{"message": "缺少 API 密鑰"}],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis,
        }
    analysis = analysis or await analyze_and_screen(user_query, selected_source["source_name"], source_id, source_type, conversation_context)
    primary_intent = max(analysis.get("intents", [{"intent": "summarize_posts", "confidence": 0.7}]), key=lambda x: x["confidence"])["intent"]
    # 內聯 get_intent_processing_params 邏輯
    intent_params = INTENT_CONFIG.get(primary_intent, INTENT_CONFIG["summarize_posts"]).get("processing", {}).copy()
    sort_override = intent_params.get("sort_override", {})
    if source_type.lower() in sort_override:
        sort = intent_params["sort"] = sort_override[source_type.lower()]
    if source_type.lower() == "lihkg" and intent_params.get("sort") == "confidence":
        sort = intent_params["sort"] = "hot"
    post_limit = max(3, min(15, analysis.get("post_limit", 5)))  # 確保 post_limit 在 3 到 15
    top_thread_ids = list(set(analysis.get("top_thread_ids", [])))
    keyword_result = await extract_keywords(user_query, conversation_context, api_key, source_type)
    max_replies = intent_params.get("max_replies", 100)
    max_comments = intent_params.get("max_replies", 100) if source_type == "reddit" else 100
    thread_data = []
    rate_limit_info = []
    processed_thread_ids = set()
    
    logger.info(f"處理查詢：查詢={user_query}, post_limit={post_limit}, 意圖={primary_intent}")
    
    try:
        if top_thread_ids and primary_intent in ["fetch_thread_by_id", "follow_up", "analyze_sentiment"]:
            for thread_id in top_thread_ids[:post_limit]:  # 尊重 post_limit
                thread_id_str = str(thread_id)
                if thread_id_str in processed_thread_ids:
                    continue
                processed_thread_ids.add(thread_id_str)
                async with cache_lock:
                    if thread_id_str in st.session_state.thread_cache and st.session_state.thread_cache[thread_id_str]["data"].get("replies"):
                        cached_data = st.session_state.thread_cache[thread_id_str]["data"]
                        thread_data.append(cached_data)
                        continue
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_thread_content(
                            thread_id=thread_id_str,
                            cat_id=source_id,
                            max_replies=max_replies,
                            fetch_last_pages=3 if keyword_result.get("time_sensitive", False) else 0
                        )
                    else:
                        result = await get_reddit_thread_content(post_id=thread_id_str, subreddit=source_id, max_comments=max_comments)
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    if result.get("title"):
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"reply in thread {thread_id_str}"),
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"] and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        total_replies = result.get("total_replies", 0)
                        if total_replies == 0 and "no_of_reply" in result:
                            total_replies = result.get("no_of_reply", 0)
                        thread_info = {
                            "thread_id": thread_id_str,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id_str}"),
                            "like_count": result.get("like_count", 0),
                            "dislike_count": result.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies),
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id_str] = {"data": thread_info, "timestamp": time.time()}
        else:
            initial_threads = []
            for page in range(1, 4):
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_topic_list(cat_id=source_id, start_page=page, max_pages=1)
                    else:
                        result = await get_reddit_topic_list(subreddit=source_id, start_page=page, max_pages=1, sort=sort)
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    items = result.get("items", [])
                    initial_threads.extend(items)
                    if len(initial_threads) >= 150:
                        initial_threads = initial_threads[:150]
                        break
                    if progress_callback:
                        progress_callback(f"已抓取第 {page}/3 頁", 0.1 + 0.2 * (page / 3), {"current_page": page, "total_pages": 3})
            filtered_items = [item for item in initial_threads if item.get("no_of_reply", 0) >= intent_params.get("min_replies", 10)]
            logger.info(f"初始帖子數量: {len(initial_threads)}, 過濾後帖子數量: {len(filtered_items)}, post_limit={post_limit}")
            for item in initial_threads:
                thread_id = str(item["thread_id"])
                async with cache_lock:
                    if thread_id not in st.session_state.thread_cache:
                        st.session_state.thread_cache[thread_id] = {
                            "data": {
                                "thread_id": thread_id,
                                "title": item["title"],
                                "no_of_reply": item.get("no_of_reply", 0),
                                "last_reply_time": unix_to_readable(item.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                                "like_count": item.get("like_count", 0),
                                "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "replies": [],
                                "fetched_pages": [],
                            },
                            "timestamp": time.time(),
                        }
            candidate_threads = []
            if primary_intent == "fetch_dates":
                candidate_threads = sorted(filtered_items, key=lambda x: x.get("last_reply_time", "1970-01-01 00:00:00"), reverse=True)[:post_limit]
            else:
                if filtered_items:
                    prioritization = await prioritize_threads_with_grok(user_query, filtered_items, selected_source["source_name"], source_id, source_type, [primary_intent], post_limit)
                    top_thread_ids = prioritization.get("top_thread_ids", [])
                    logger.info(f"優先排序結果: top_thread_ids={top_thread_ids}, 原因={prioritization.get('reason', '無原因')}, 最終選取數量={len(top_thread_ids)}")
                    valid_thread_ids = [tid for tid in top_thread_ids if str(tid) in [str(item["thread_id"]) for item in filtered_items]]
                    logger.info(f"驗證後的帖子 ID: valid_thread_ids={valid_thread_ids}, 原始 top_thread_ids={top_thread_ids}")
                    candidate_threads = [item for item in filtered_items if str(item["thread_id"]) in valid_thread_ids][:post_limit]
                    if not candidate_threads:
                        logger.warning(f"未找到匹配的帖子 ID: {top_thread_ids}, 回退到按熱門度排序")
                        candidate_threads = sorted(filtered_items, key=lambda x: x.get("no_of_reply", 0) * 0.6 + x.get("like_count", 0) * 0.4, reverse=True)[:post_limit]
            logger.info(f"候選帖子: 數量={len(candidate_threads)}, 帖子 ID={[item['thread_id'] for item in candidate_threads]}, post_limit={post_limit}")
            if not candidate_threads:
                logger.warning(f"未找到候選帖子: 查詢={user_query}, 來源={selected_source}")
                return {
                    "selected_source": selected_source,
                    "thread_data": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": rate_limit_until,
                    "analysis": analysis,
                }
            if progress_callback:
                progress_callback("正在抓取帖子內容", 0.3)
            for item in candidate_threads:
                thread_id = str(item["thread_id"])
                if thread_id in processed_thread_ids:
                    continue
                processed_thread_ids.add(thread_id)
                async with cache_lock:
                    if thread_id in st.session_state.thread_cache and st.session_state.thread_cache[thread_id]["data"].get("replies"):
                        cached_data = st.session_state.thread_cache[thread_id]["data"]
                        thread_data.append(cached_data)
                        continue
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_thread_content(
                            thread_id=thread_id,
                            cat_id=source_id,
                            max_replies=max_replies,
                            fetch_last_pages=3 if keyword_result.get("time_sensitive", False) else 0
                        )
                    else:
                        result = await get_reddit_thread_content(post_id=thread_id, subreddit=source_id, max_comments=max_comments)
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    if result.get("title"):
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"reply in thread {thread_id}"),
                            }
                            for reply in result.get("replies", [])
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"] and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        total_replies = result.get("total_replies", item.get("no_of_reply", 0))
                        if total_replies == 0:
                            total_replies = item.get("no_of_reply", 0)
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies),
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}
        if len(thread_data) < post_limit and primary_intent == "follow_up":
            supplemental_result = await (get_lihkg_topic_list(cat_id=source_id, start_page=1, max_pages=2) if source_type == "lihkg" else get_reddit_topic_list(subreddit=source_id, start_page=1, max_pages=2, sort=sort))
            supplemental_threads = [item for item in supplemental_result.get("items", []) if str(item["thread_id"]) not in top_thread_ids and any(kw.lower() in item["title"].lower() for kw in keyword_result.get("keywords", ["新話題"]))][:post_limit - len(thread_data)]  # 尊重 post_limit
            logger.info(f"補充帖子: 數量={len(supplemental_threads)}, 帖子 ID={[item['thread_id'] for item in supplemental_threads]}")
            for item in supplemental_threads:
                thread_id = str(item["thread_id"])
                if thread_id in processed_thread_ids:
                    continue
                processed_thread_ids.add(thread_id)
                async with request_semaphore:
                    if source_type == "lihkg":
                        result = await get_lihkg_thread_content(
                            thread_id=thread_id,
                            cat_id=source_id,
                            max_replies=max_replies,
                            fetch_last_pages=3 if keyword_result.get("time_sensitive", False) else 0
                        )
                    else:
                        result = await get_reddit_thread_content(post_id=thread_id, subreddit=source_id, max_comments=max_comments)
                    request_counter = result.get("request_counter", request_counter)
                    last_reset = result.get("last_reset", last_reset)
                    rate_limit_until = result.get("rate_limit_until", rate_limit_until)
                    rate_limit_info.extend(result.get("rate_limit_info", []))
                    if result.get("title"):
                        filtered_replies = [
                            {
                                "reply_id": reply.get("reply_id"),
                                "msg": clean_html(reply.get("msg", "[無內容]")),
                                "like_count": reply.get("like_count", 0),
                                "dislike_count": reply.get("dislike_count", 0) if source_type == "lihkg" else 0,
                                "reply_time": unix_to_readable(reply.get("reply_time", "0"), context=f"supplemental reply in thread {thread_id}"),
                            }
                            for reply in result.get("replies", []) 
                            if reply.get("msg") and clean_html(reply.get("msg")) not in ["[無內容]", "[圖片]", "[表情符號]"] and len(clean_html(reply.get("msg")).strip()) > 7
                        ]
                        total_replies = result.get("total_replies", item.get("no_of_reply", 0))
                        if total_replies == 0:
                            total_replies = item.get("no_of_reply", 0)
                        thread_info = {
                            "thread_id": thread_id,
                            "title": result.get("title"),
                            "no_of_reply": total_replies,
                            "last_reply_time": unix_to_readable(result.get("last_reply_time", "0"), context=f"thread {thread_id}"),
                            "like_count": item.get("like_count", 0),
                            "dislike_count": item.get("dislike_count", 0) if source_type == "lihkg" else 0,
                            "replies": filtered_replies,
                            "fetched_pages": result.get("fetched_pages", []),
                            "total_fetched_replies": len(filtered_replies),
                        }
                        thread_data.append(thread_info)
                        async with cache_lock:
                            st.session_state.thread_cache[thread_id] = {"data": thread_info, "timestamp": time.time()}
        if progress_callback:
            progress_callback("正在準備數據", 0.5)
        logger.info(f"最終 thread_data：數量={len(thread_data)}, 帖子 ID={[d['thread_id'] for d in thread_data]}, post_limit={post_limit}, 來源={selected_source}")
        return {
            "selected_source": selected_source,
            "thread_data": thread_data,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            " humanities": [],
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis,
        }
    except Exception as e:
        logger.error(
            f"查詢處理錯誤：查詢={user_query}, 錯誤={str(e)}, "
            f"filtered_items_count={len(filtered_items) if 'filtered_items' in locals() else 0}, "
            f"top_thread_ids={top_thread_ids if 'top_thread_ids' in locals() else []}, "
            f"valid_thread_ids={valid_thread_ids if 'valid_thread_ids' in locals() else []}, "
            f"candidate_threads_count={len(candidate_threads) if 'candidate_threads' in locals() else 0}"
        )
        return {
            "selected_source": selected_source,
            "thread_data": [],
            "rate_limit_info": rate_limit_info if 'rate_limit_info' in locals() else [],
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": rate_limit_until,
            "analysis": analysis,
            "error": str(e),
        }

def clean_cache(max_age=3600):
    current_time = time.time()
    expired_keys = [key for key, value in st.session_state.thread_cache.items() if current_time - value["timestamp"] > max_age]
    for key in expired_keys:
        del st.session_state.thread_cache[key]
    if len(st.session_state.thread_cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(st.session_state.thread_cache.items(), key=lambda x: x[1]["timestamp"])[:len(st.session_state.thread_cache) - MAX_CACHE_SIZE]
        for key, _ in sorted_keys:
            del st.session_state.thread_cache[key]
        logger.info(f"清理緩存，移除 {len(sorted_keys)} 個條目，當前大小：{len(st.session_state.thread_cache)}")

def configure_lihkg_api_logger():
    configure_logger("lihkg_api", "lihkg_api.log")

def configure_reddit_api_logger():
    configure_logger("reddit_api", "reddit_api.log")