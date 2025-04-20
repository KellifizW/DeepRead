"""
Grok 3 API 處理模組，負責問題分析、帖子篩選和回應生成。
包含數據處理邏輯（進階分析、緩存管理）和輔助函數。
主要函數：
- analyze_and_screen：分析問題並篩選帖子。
- stream_grok3_response：生成流式回應。
- process_user_question：處理用戶問題，抓取並分析帖子。
- clean_html：清理 HTML 標籤。
硬編碼參數（優化建議：移至配置文件或介面）：
- post_limit=2, max=20
- reply_limit=75/200, max_replies=25
- min_replies=20/50, min_likes=10/20
- target_pages=60%
- replies_filter=25
- max_tokens=600/1000
- temperature=0.7
- timeout=30
- GROK3_TOKEN_LIMIT=100000
- retries=3
- retry_sleep=2^attempt
- fetch_pages=1-3
- sleep=1
"""

import aiohttp
import asyncio
import json
import re
import random
import math
import time
import logging
import streamlit as st
from lihkg_api import get_lihkg_topic_list, get_lihkg_thread_content

logger = logging.getLogger(__name__)
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"
GROK3_TOKEN_LIMIT = 100000  # 硬編碼：建議可配置

def clean_html(text):
    clean = re.compile(r'<[^>]+>')
    text = clean.sub('', text)
    return re.sub(r'\s+', ' ', text).strip()

async def analyze_and_screen(user_query, cat_name, cat_id, thread_titles=None, metadata=None, thread_data=None, is_advanced=False):
    prompt = f"""
    你是一個智能助手，分析用戶問題並篩選 LIHKG 帖子。以繁體中文回覆，輸出 JSON。

    問題：{user_query}
    分類：{cat_name}（cat_id={cat_id})
    {'帖子標題：' + json.dumps(thread_titles, ensure_ascii=False) if thread_titles else ''}
    {'元數據：' + json.dumps(metadata, ensure_ascii=False) if metadata else ''}
    {'回覆數據：' + json.dumps(thread_data, ensure_ascii=False) if thread_data else ''}

    步驟：
    1. 識別主題（感動、搞笑、財經等），標記為 theme。
    2. 判斷意圖（總結、情緒分析、幽默總結）。
    3. {'檢查帖子是否達60%頁數（總頁數*0.6，向上取整），若未達標，設置 needs_advanced_analysis=True。' if is_advanced else '篩選帖子：'}
       {'- 若無標題，設置初始抓取（30-90個標題）。' if not thread_titles else '- 從標題選20個候選（candidate_thread_ids），再選top_thread_ids，確保每個ID唯一。'}
    4. 設置參數：
       - theme：問題主題。
       - category_ids：[cat_id]。
       - data_type："title"、"replies"、"both"。
       - post_limit：從問題提取（默認2，最大20，根據問題複雜度動態選擇1-20）。
       - reply_limit：{200 if is_advanced else 75}。
       - filters：根據主題（感動：like_count≥5；搞笑：like_count≥10；財經：like_count≥10；其他：min_replies≥20，min_likes≥10）。
       - processing：emotion_focused_summary、humor_focused_summary、professional_summary、summarize、sentiment。
       - candidate_thread_ids：20個候選ID，確保唯一。
       - top_thread_ids：最終選定ID，確保唯一。
    5. 若無關LIHKG，返回空category_ids。

    輸出：
    {"{ \"needs_advanced_analysis\": false, \"suggestions\": { \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [] }, \"reason\": \"\" }" if is_advanced else "{ \"theme\": \"\", \"category_ids\": [], \"data_type\": \"\", \"post_limit\": 0, \"reply_limit\": 0, \"filters\": {}, \"processing\": \"\", \"candidate_thread_ids\": [], \"top_thread_ids\": [], \"category_suggestion\": \"\" }"}
    """
    
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        logger.error("Grok 3 API key missing")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 200 if is_advanced else 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": "Missing API key"
        } if not is_advanced else {
            "needs_advanced_analysis": False, "suggestions": {}, "reason": "Missing API key"
        }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 600,  # 硬編碼：建議可配置
        "temperature": 0.7  # 硬編碼：建議可配置
    }
    
    try:
        logger.info(f"Grok 3 API request: url={GROK3_API_URL}, prompt_length={len(prompt)}, prompt_summary={prompt[:50]}...")
        async with aiohttp.ClientSession() as session:
            async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:  # 硬編碼：timeout=30
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                logger.info(f"Grok 3 API response: status={response.status}, response_summary={str(data)[:50]}...")
                if is_advanced:
                    result["suggestions"]["category_ids"] = [cat_id]
                else:
                    result["category_ids"] = [cat_id]
                return result
    except Exception as e:
        logger.error(f"Grok 3 API failed: error={str(e)}, prompt_summary={prompt[:50]}...")
        return {
            "theme": "未知", "category_ids": [cat_id], "data_type": "both", "post_limit": 2,
            "reply_limit": 200 if is_advanced else 75, "filters": {"min_replies": 20, "min_likes": 10},
            "processing": "summarize", "candidate_thread_ids": [], "top_thread_ids": [],
            "category_suggestion": f"Analysis failed: {str(e)}"
        } if not is_advanced else {
            "needs_advanced_analysis": False, "suggestions": {"category_ids": [cat_id]},
            "reason": f"Analysis failed: {str(e)}"
        }

async def stream_grok3_response(user_query, metadata, thread_data, processing):
    try:
        GROK3_API_KEY = st.secrets["grok3key"]
    except KeyError:
        yield "錯誤: 缺少 API 密鑰"
        return
    
    filtered_thread_data = {
        tid: {
            "thread_id": data["thread_id"], "title": data["title"], "no_of_reply": data.get("no_of_reply", 0),
            "last_reply_time": data.get("last_reply_time", 0), "like_count": data.get("like_count", 0),
            "dislike_count": data.get("dislike_count", 0),
            "replies": [r for r in data.get("replies", []) if r.get("like_count", 0) != 0 or r.get("dislike_count", 0) != 0][:25],  # 硬編碼：建議可配置
            "fetched_pages": data.get("fetched_pages", [])
        } for tid, data in thread_data.items()
    }
    
    needs_advanced_analysis = False
    reason = ""
    for data in filtered_thread_data.values():
        total_pages = (data["no_of_reply"] + 24) // 25
        target_pages = math.ceil(total_pages * 0.6)  # 硬編碼：建議可配置
        if len(data["fetched_pages"]) < target_pages:
            needs_advanced_analysis = True
            reason += f"帖子 {data['thread_id']} 僅抓取 {len(data['fetched_pages'])}/{total_pages} 頁，未達60%。"
    
    prompt_templates = {
        "emotion_focused_summary": f"""
        總結 LIHKG 感動或溫馨帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦感動情緒，引用高關注回覆，適配分類語氣（吹水台輕鬆，創意台溫馨）。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "humor_focused_summary": f"""
        總結 LIHKG 幽默或搞笑帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦幽默情緒，引用高關注回覆，適配分類語氣（吹水台輕鬆，成人台大膽）。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "professional_summary": f"""
        總結 LIHKG 財經或時事帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        聚焦專業觀點，引用高關注回覆，適配分類語氣（財經台專業，時事台嚴肅）。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "summarize": f"""
        總結 LIHKG 帖子，300-500字。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        引用高關注回覆，適配分類語氣。
        輸出：總結\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """,
        "sentiment": f"""
        分析 LIHKG 帖子情緒。問題：{user_query}
        帖子：{json.dumps(metadata, ensure_ascii=False)}
        回覆：{json.dumps(filtered_thread_data, ensure_ascii=False)}
        判斷情緒分佈（正面、負面、中立），聚焦高關注回覆。
        輸出：情緒分析：正面XX%，負面XX%，中立XX%\n依據：...\n進階分析建議：needs_advanced_analysis={needs_advanced_analysis}, reason={reason}
        """
    }
    
    prompt = prompt_templates.get(processing, f"直接回答問題，50-100字。問題：{user_query}\n輸出：回應")
    if len(prompt) > GROK3_TOKEN_LIMIT:
        for tid in filtered_thread_data:
            filtered_thread_data[tid]["replies"] = filtered_thread_data[tid]["replies"][:10]
        prompt = prompt.replace(json.dumps(filtered_thread_data, ensure_ascii=False), json.dumps(filtered_thread_data, ensure_ascii=False))
        logger.info(f"Truncated prompt: length={len(prompt)}")
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {GROK3_API_KEY}"}
    payload = {
        "model": "grok-3-beta",
        "messages": [{"role": "system", "content": "以繁體中文回答，僅基於提供數據。"}, {"role": "user", "content": prompt}],
        "max_tokens": 1000,  # 硬編碼：建議可配置
        "temperature": 0.7,  # 硬編碼：建議可配置
        "stream": True
    }
    
    for attempt in range(3):  # 硬編碼：建議可配置
        try:
            logger.info(f"Grok 3 API request: url={GROK3_API_URL}, attempt={attempt+1}, prompt_length={len(prompt)}, prompt_summary={prompt[:50]}...")
            async with aiohttp.ClientSession() as session:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=30) as response:  # 硬編碼：timeout=30
                    async for line in response.content:
                        if line and not line.isspace():
                            line_str = line.decode('utf-8').strip()
                            if line_str == "data: [DONE]":
                                break
                            if line_str.startswith("data: "):
                                try:
                                    chunk = json.loads(line_str[6:])
                                    content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                                    if content and "進階分析建議：" not in content:
                                        yield content
                                except json.JSONDecodeError as e:
                                    logger.warning(f"JSON decode error: {str(e)}, line={line_str}")
                                    continue
                    logger.info(f"Grok 3 API response: status={response.status}, stream_completed=True")
                    return
        except Exception as e:
            logger.error(f"Grok 3 API failed: attempt={attempt+1}, error={str(e)}, prompt_summary={prompt[:50]}...")
            if attempt < 2:
                await asyncio.sleep(2 ** attempt)  # 硬編碼：建議可配置
                continue
            yield f"錯誤: 連線失敗，請稍後重試"
            return

async def process_user_question(user_question, selected_cat, cat_id, analysis, request_counter, last_reset, rate_limit_until, is_advanced=False, previous_thread_ids=None, previous_thread_data=None):
    post_limit = min(analysis.get("post_limit", 2), 20)  # 硬編碼：最大20，Grok3動態選擇
    reply_limit = 200 if is_advanced else min(analysis.get("reply_limit", 75), 75)  # 硬編碼：建議可配置
    filters = analysis.get("filters", {})
    min_replies = 20 if analysis.get("theme") == "搞笑" else filters.get("min_replies", 50)  # 硬編碼：建議可配置
    min_likes = 10 if analysis.get("theme") == "搞笑" else filters.get("min_likes", 20)  # 硬編碼：建議可配置
    candidate_thread_ids = list(set(analysis.get("candidate_thread_ids", [])))  # 確保唯一
    top_thread_ids = list(set(analysis.get("top_thread_ids", []))) if not is_advanced else (list(set(previous_thread_ids or [])))
    
    thread_data = []
    rate_limit_info = []
    
    if is_advanced and top_thread_ids:
        logger.info(f"Advanced analysis: thread_ids={top_thread_ids}, previous_data={bool(previous_thread_data)}")
        for thread_id in top_thread_ids:
            cached_data = previous_thread_data.get(thread_id) if previous_thread_data else None
            fetched_pages = cached_data.get("fetched_pages", []) if cached_data else []
            existing_replies = cached_data.get("replies", []) if cached_data else []
            total_replies = cached_data.get("no_of_reply", 0) if cached_data else 0
            
            total_pages = (total_replies + 24) // 25
            target_pages = math.ceil(total_pages * 0.6)  # 硬編碼：建議可配置
            remaining_pages = max(0, target_pages - len(fetched_pages))
            
            if remaining_pages <= 0:
                logger.info(f"Thread {thread_id} meets 60% page threshold: {len(fetched_pages)}/{target_pages}")
                thread_data.append({
                    "thread_id": str(thread_id), "title": cached_data.get("title", "未知標題"),
                    "no_of_reply": total_replies, "last_reply_time": cached_data.get("last_reply_time", 0),
                    "like_count": cached_data.get("like_count", 0), "dislike_count": cached_data.get("dislike_count", 0),
                    "replies": existing_replies, "fetched_pages": fetched_pages
                })
                continue
            
            logger.info(f"Fetching remaining pages for thread_id={thread_id}: remaining={remaining_pages}")
            start_page = max(fetched_pages, default=1) + 1 if fetched_pages else 1
            thread_result = await get_lihkg_thread_content(
                thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
                rate_limit_until=rate_limit_until, max_replies=reply_limit, fetch_last_pages=remaining_pages, start_page=start_page
            )
            
            request_counter = thread_result.get("request_counter", request_counter)
            last_reset = thread_result.get("last_reset", last_reset)
            rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
            rate_limit_info.extend(thread_result.get("rate_limit_info", []))
            
            replies = thread_result.get("replies", [])
            if not replies and thread_result.get("total_replies", 0) >= min_replies:
                logger.warning(f"Invalid thread: {thread_id}")
                continue
            
            all_replies = existing_replies + [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in replies]
            sorted_replies = sorted(all_replies, key=lambda x: x.get("like_count", 0), reverse=True)[:reply_limit]
            all_fetched_pages = sorted(set(fetched_pages + thread_result.get("fetched_pages", [])))
            
            thread_data.append({
                "thread_id": str(thread_id), "title": thread_result.get("title", cached_data.get("title", "未知標題") if cached_data else "未知標題"),
                "no_of_reply": thread_result.get("total_replies", total_replies), "last_reply_time": thread_result.get("last_reply_time", cached_data.get("last_reply_time", 0) if cached_data else 0),
                "like_count": thread_result.get("like_count", cached_data.get("like_count", 0) if cached_data else 0),
                "dislike_count": thread_result.get("dislike_count", cached_data.get("dislike_count", 0) if cached_data else 0),
                "replies": sorted_replies, "fetched_pages": all_fetched_pages
            })
            logger.info(f"Advanced thread {thread_id}: replies={len(sorted_replies)}, pages={len(all_fetched_pages)}/{target_pages}")
            await asyncio.sleep(1)  # 硬編碼：建議可配置
        
        return {
            "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    initial_threads = []
    for page in range(1, 4):  # 硬編碼：建議可配置
        result = await get_lihkg_topic_list(cat_id=cat_id, start_page=page, max_pages=1, request_counter=request_counter, last_reset=last_reset, rate_limit_until=rate_limit_until)
        request_counter = result.get("request_counter", request_counter)
        last_reset = result.get("last_reset", last_reset)
        rate_limit_until = result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(result.get("rate_limit_info", []))
        initial_threads.extend(result.get("items", []))
        logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(result.get('items', []))}")
        if len(initial_threads) >= 90:
            initial_threads = initial_threads[:90]
            break
    
    filtered_items = [
        item for item in initial_threads
        if item.get("no_of_reply", 0) >= min_replies and int(item.get("like_count", 0)) >= min_likes
    ]
    logger.info(f"Filtered items: {len(filtered_items)} from {len(initial_threads)}")
    
    for item in initial_threads:
        thread_id = str(item["thread_id"])
        st.session_state.thread_cache[thread_id] = {
            "data": {
                "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
                "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
                "dislike_count": item.get("dislike_count", 0), "replies": [], "fetched_pages": []
            },
            "timestamp": time.time()
        }
    
    analysis = await analyze_and_screen(user_query=user_question, cat_name=selected_cat, cat_id=cat_id, thread_titles=filtered_items[:90], metadata=None, thread_data=None)
    top_thread_ids = list(set(analysis.get("top_thread_ids", [])))  # 確保唯一
    if not top_thread_ids and filtered_items:
        top_thread_ids = list(set([item["thread_id"] for item in random.sample(filtered_items, min(post_limit, len(filtered_items)))]))  # 修復：補全random.sample
        logger.warning(f"No top_thread_ids, randomly selected: {top_thread_ids}")
    
    seen_ids = set()
    candidate_threads = []
    for item in filtered_items:
        if str(item["thread_id"]) in map(str, top_thread_ids) and item["thread_id"] not in seen_ids:
            candidate_threads.append(item)
            seen_ids.add(item["thread_id"])
            if len(candidate_threads) >= post_limit:
                break
    
    if not candidate_threads and filtered_items:
        candidate_threads = random.sample(filtered_items, min(post_limit, len(filtered_items)))
        logger.info(f"No candidate threads, using random: {len(candidate_threads)}")
    
    logger.info(f"Selected candidate threads: thread_ids={[item['thread_id'] for item in candidate_threads]}")
    
    for item in candidate_threads:
        thread_id = str(item["thread_id"])
        thread_result = await get_lihkg_thread_content(
            thread_id=thread_id, cat_id=cat_id, request_counter=request_counter, last_reset=last_reset,
            rate_limit_until=rate_limit_until, max_replies=25, fetch_last_pages=0  # 硬編碼：建議可配置
        )
        request_counter = thread_result.get("request_counter", request_counter)
        last_reset = thread_result.get("last_reset", last_reset)
        rate_limit_until = thread_result.get("rate_limit_until", rate_limit_until)
        rate_limit_info.extend(thread_result.get("rate_limit_info", []))
        
        replies = thread_result.get("replies", [])
        if not replies and thread_result.get("total_replies", 0) >= min_replies:
            logger.warning(f"Invalid thread: {thread_id}")
            continue
        
        sorted_replies = sorted(replies, key=lambda x: x.get("like_count", 0), reverse=True)[:25]  # 硬編碼：建議可配置
        thread_data.append({
            "thread_id": thread_id, "title": item["title"], "no_of_reply": item.get("no_of_reply", 0),
            "last_reply_time": item.get("last_reply_time", 0), "like_count": item.get("like_count", 0),
            "dislike_count": item.get("dislike_count", 0),
            "replies": [{"msg": clean_html(r["msg"]), "like_count": r.get("like_count", 0), "dislike_count": r.get("dislike_count", 0), "reply_time": r.get("reply_time", 0)} for r in sorted_replies],
            "fetched_pages": thread_result.get("fetched_pages", [1])
        })
        st.session_state.thread_cache[thread_id]["data"].update({
            "replies": thread_data[-1]["replies"], "fetched_pages": thread_data[-1]["fetched_pages"]
        })
    
    return {
        "selected_cat": selected_cat, "thread_data": thread_data, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
    }
