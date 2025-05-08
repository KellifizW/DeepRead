import json
import logging
import re
import aiohttp
import asyncio
from logging_config import configure_logger
import uuid

# 配置日誌
logger = configure_logger(__name__, "dynamic_prompt_utils.log")

# 配置參數
CONFIG = {
    "max_prompt_length": 120000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "default_intents": ["recommend_threads", "summarize_posts", "time_sensitive_analysis", "rank_topics"],
    "error_prompt_template": "在 {selected_cat} 中未找到符合條件的帖子（篩選：{filters}）。建議嘗試其他關鍵詞或討論區！",
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
    "default_word_ranges": {
        "summarize_posts": (700, 1500),
        "analyze_sentiment": (700, 1500),
        "follow_up": (1000, 2500),
        "fetch_thread_by_id": (500, 1000),
        "general_query": (500, 1000),
        "list_titles": (500, 1500),
        "find_themed": (700, 1500),
        "fetch_dates": (500, 1000),
        "search_keywords": (700, 1500),
        "recommend_threads": (500, 1500),
        "time_sensitive_analysis": (500, 1000),
        "contextual_analysis": (700, 1500),
        "rank_topics": (500, 1000)
    }
}

async def parse_query(query, conversation_context, grok3_api_key, source_type="lihkg"):
    conversation_context = conversation_context or []
    
    # 驗證輸入參數
    if not isinstance(query, str):
        logger.error(f"無效查詢類型：預期 str，得到 {type(query)}")
        return {
            "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "無效查詢類型"}],
            "keywords": [],
            "related_terms": [],
            "time_range": "all",
            "thread_ids": [],
            "reason": "無效查詢類型",
            "confidence": 0.5
        }
    
    # 提取關鍵詞和相關詞
    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords = keyword_result.get("keywords", [])
    related_terms = keyword_result.get("related_terms", [])
    time_sensitive = keyword_result.get("time_sensitive", False)
    
    # 檢查是否為特定帖子 ID 查詢
    id_match = re.search(r'(?:ID|帖子)\s*([a-zA-Z0-9]+)', query, re.IGNORECASE)
    if id_match:
        thread_id = id_match.group(1)
        logger.info(f"檢測到特定帖子 ID 查詢：thread_id={thread_id}")
        return {
            "intents": [{"intent": "fetch_thread_by_id", "confidence": 0.95, "reason": f"檢測到明確帖子 ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到明確帖子 ID {thread_id}",
            "confidence": 0.95
        }
    
    # 檢查是否為追問
    thread_id, _, _, match_reason = await extract_relevant_thread(
        conversation_context, query, grok3_api_key
    )
    if thread_id:
        logger.info(f"檢測到追問：thread_id={thread_id}, 原因={match_reason}")
        return {
            "intents": [{"intent": "follow_up", "confidence": 0.90, "reason": f"檢測到追問，匹配帖子 ID {thread_id}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [thread_id],
            "reason": f"檢測到追問，匹配帖子 ID {thread_id}，原因：{match_reason}",
            "confidence": 0.90
        }
    
    # 檢查是否為情緒分析查詢
    sentiment_triggers = ["情緒", "氣氛", "指數", "正負面", "sentiment", "mood"]
    is_sentiment_query = any(word in query for word in sentiment_triggers)
    if is_sentiment_query:
        logger.info(f"檢測到情緒分析查詢：query={query}")
        intents = [
            {"intent": "analyze_sentiment", "confidence": 0.90, "reason": "檢測到情緒分析關鍵詞，需分析情緒比例"},
            {"intent": "summarize_posts", "confidence": 0.80, "reason": "情緒分析查詢，輔以總結"}
        ]
        time_range = "recent" if time_sensitive else "all"
        return {
            "intents": intents,
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": time_range,
            "thread_ids": [],
            "reason": "檢測到情緒分析查詢，優先分析情緒比例",
            "confidence": 0.90
        }
    
    # 檢查是否為時間敏感查詢
    time_triggers = ["今晚", "今日", "最近", "今個星期"]
    is_time_sensitive = time_sensitive or any(word in query for word in time_triggers)
    if is_time_sensitive:
        logger.info(f"檢測到時間敏感查詢：query={query}")
        return {
            "intents": [
                {"intent": "time_sensitive_analysis", "confidence": 0.85, "reason": "時間敏感查詢，需優先最新帖子"},
                {"intent": "summarize_posts", "confidence": 0.80, "reason": "時間敏感查詢，輔以總結"}
            ],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "recent",
            "thread_ids": [],
            "reason": "檢測到時間敏感查詢，優先最新帖子",
            "confidence": 0.85
        }
    
    # 檢查是否為標題列舉查詢
    trigger_words = ["列出", "標題", "清單", "所有標題"]
    detected_triggers = [word for word in trigger_words if word in query]
    if detected_triggers:
        logger.info(f"檢測到 list_titles 觸發詞：{detected_triggers}")
        return {
            "intents": [{"intent": "list_titles", "confidence": 0.95, "reason": f"檢測到 list_titles 觸發詞：{detected_triggers}"}],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [],
            "reason": f"檢測到 list_titles 觸發詞：{detected_triggers}",
            "confidence": 0.95
        }
    
    # 檢查是否為排序查詢
    ranking_triggers = ["熱門", "最多", "關注", "流行"]
    is_ranking_query = any(word in query for word in ranking_triggers)
    if is_ranking_query:
        logger.info(f"檢測到排序查詢：query={query}")
        intents = [
            {"intent": "rank_topics", "confidence": 0.85, "reason": "檢測到排序關鍵詞，需按關注度排序話題"},
            {"intent": "summarize_posts", "confidence": 0.80, "reason": "排序查詢，輔以總結"}
        ]
        time_range = "recent" if time_sensitive else "all"
        return {
            "intents": intents,
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": time_range,
            "thread_ids": [],
            "reason": "檢測到排序查詢，優先排序話題",
            "confidence": 0.85
        }
    
    # 檢查平台特定查詢
    platform_triggers = ["Reddit", "LIHKG", "子版", "討論區"]
    is_platform_specific = any(word in query for word in platform_triggers)
    if is_platform_specific:
        logger.info(f"檢測到平台特定查詢：query={query}")
        return {
            "intents": [
                {"intent": "contextual_analysis", "confidence": 0.85, "reason": "平台特定查詢，需語義和情緒分析"},
                {"intent": "summarize_posts", "confidence": 0.80, "reason": "平台特定查詢，輔以總結"}
            ],
            "keywords": keywords,
            "related_terms": related_terms,
            "time_range": "all",
            "thread_ids": [],
            "reason": "檢測到平台特定查詢，優先語義分析",
            "confidence": 0.85
        }
    
    # 語義意圖分析
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"])
    multi_intent_indicators = ["並且", "同時", "總結並", "列出並", "分析並"]
    has_multi_intent = any(indicator in query for indicator in multi_intent_indicators)
    
    prompt = f"""
你是一個語義分析助手，請分析以下查詢並分類最多4個意圖，考慮對話歷史、關鍵詞和相關詞。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(keywords, ensure_ascii=False)}
相關詞：{json.dumps(related_terms, ensure_ascii=False)}
數據來源：{source_type}
是否模糊查詢：{is_vague}
是否包含多意圖指示詞：{has_multi_intent}
特別注意：
- 支持粵語及口語化表達（如「既」「係咩」）。
意圖描述：
{json.dumps({
    "list_titles": "列出帖子標題或清單（觸發詞：列出、標題、清單、所有標題）",
    "summarize_posts": "總結帖子內容或討論",
    "analyze_sentiment": "分析帖子或回覆的情緒（觸發詞：情緒、氣氛、指數、正負面、sentiment、mood）",
    "general_query": "模糊或非討論區相關問題",
    "find_themed": "尋找特定主題的帖子",
    "fetch_dates": "提取帖子或回覆的日期資料",
    "search_keywords": "根據關鍵詞搜索帖子",
    "recommend_threads": "推薦相關或熱門帖子",
    "follow_up": "追問之前回應的帖子內容",
    "fetch_thread_by_id": "根據明確的帖子 ID 抓取內容",
    "time_sensitive_analysis": "處理時間敏感查詢，優先最新帖子",
    "contextual_analysis": "綜合總結帖子觀點、推薦相關帖子、分析情緒（適用於平台特定查詢）",
    "rank_topics": "按關注度排序話題或帖子（觸發詞：熱門、最多、關注）"
}, ensure_ascii=False, indent=2)}
輸出格式：{{
  "intents": [
    {{"intent": "意圖1", "confidence": 0.0-1.0, "reason": "匹配原因"}},
    ...
  ],
  "reason": "整體匹配原因"
}}
若查詢包含「列出」「標題」「清單」「所有標題」，優先返回 list_titles 意圖，信心值設為 0.95。
若查詢包含「情緒」「氣氛」「指數」「正負面」「sentiment」「mood」，優先 analyze_sentiment 意圖。
若查詢包含「Reddit」「LIHKG」「子版」「討論區」，優先 contextual_analysis 意圖。
若查詢包含時間敏感詞（如「今晚」「今日」），優先 time_sensitive_analysis 意圖。
若查詢包含「熱門」「最多」「關注」，優先 rank_topics 意圖。
若查詢模糊且無多意圖指示詞，僅返回1個高信心意圖（優先 list_titles 若包含觸發詞，否則 rank_topics 或 summarize_posts）。
僅返回信心值高於 {CONFIG['intent_confidence_threshold']} 的意圖。
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，以繁體中文回答，專注於理解用戶意圖。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.5
    }
    
    for attempt in range(CONFIG["max_parse_retries"]):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=CONFIG["parse_timeout"]
                ) as response:
                    if response.status != 200:
                        logger.debug(f"意圖分析失敗：狀態碼={response.status}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"意圖分析失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    intents = result.get("intents", [{"intent": "summarize_posts", "confidence": 0.7, "reason": "模糊查詢，默認總結帖子"}])
                    reason = result.get("reason", "語義匹配")
                    
                    # 過濾低信心意圖
                    intents = [i for i in intents if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                    if not intents:
                        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "無高信心意圖，默認總結"}]
                    
                    # 對於模糊查詢，限制為單一意圖
                    if is_vague and not has_multi_intent:
                        intents = [max(intents, key=lambda x: x["confidence"])]
                    
                    time_range = "recent" if is_time_sensitive else "all"
                    logger.info(f"意圖分析完成：intents={[i['intent'] for i in intents]}, reason={reason}, confidence={max(i['confidence'] for i in intents)}")
                    return {
                        "intents": intents[:4],
                        "keywords": keywords,
                        "related_terms": related_terms,
                        "time_range": time_range,
                        "thread_ids": [],
                        "reason": reason,
                        "confidence": max(intent["confidence"] for intent in intents) if intents else 0.7
                    }
        except Exception as e:
            logger.debug(f"意圖分析錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    # 回退到默認意圖
    logger.warning(f"意圖分析失敗，回退到默認意圖：summarize_posts")
    return {
        "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "意圖分析失敗，默認總結帖子"}],
        "keywords": keywords,
        "related_terms": related_terms,
        "time_range": "all",
        "thread_ids": [],
        "reason": "意圖分析失敗，默認總結帖子",
        "confidence": 0.5
    }

async def extract_keywords(query, conversation_context, grok3_api_key, source_type="lihkg"):
    """
    提取查詢中的語義關鍵詞和相關詞，考慮 Reddit 和 LIHKG 上下文。
    """
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    
    platform_terms = {
        "lihkg": ["吹水", "高登", "連登"],
        "reddit": ["YOLO", "DD", "subreddit"]
    }
    
    prompt = f"""
請從以下查詢提取 {CONFIG['min_keywords']}-{CONFIG['max_keywords']} 個核心關鍵詞（優先名詞或核心動詞，排除通用詞如 {generic_terms}）。
同時生成最多8個語義相關詞（同義詞或討論區術語，如 {platform_terms[source_type]}）。
若查詢包含時間性詞語（如「今晚」「今日」「最近」），設置 time_sensitive 為 true。
若查詢模糊（如「有咩post 分享」），根據對話歷史推測語義意圖。
若查詢包含粵語表達（如「最多」「既」），保留排序或疑問相關語義。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
數據來源：{source_type}
返回格式：
{{
  "keywords": ["關鍵詞1", "關鍵詞2", ...],
  "related_terms": ["相關詞1", "相關詞2", ...],
  "reason": "提取邏輯（70字以內）",
  "time_sensitive": true/false
}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": f"你是語義分析助手，以繁體中文回答，專注於提取 {source_type} 討論區關鍵詞。"},
            *conversation_context,
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }
    
    for attempt in range(CONFIG["max_parse_retries"]):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=CONFIG["parse_timeout"]
                ) as response:
                    if response.status != 200:
                        logger.debug(f"關鍵詞提取失敗：狀態碼={response.status}，嘗試次數={attempt + 1}")
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.debug(f"關鍵詞提取失敗：缺少 choices，嘗試次數={attempt + 1}")
                        continue
                    result = json.loads(data["choices"][0]["message"]["content"])
                    keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
                    related_terms = result.get("related_terms", [])
                    return {
                        "keywords": keywords[:CONFIG["max_keywords"]],
                        "related_terms": related_terms[:8],
                        "reason": result.get("reason", "未提供原因")[:70],
                        "time_sensitive": result.get("time_sensitive", False)
                    }
        except Exception as e:
            logger.debug(f"關鍵詞提取錯誤：{str(e)}，嘗試次數={attempt + 1}")
            if attempt < CONFIG["max_parse_retries"] - 1:
                await asyncio.sleep(2)
            continue
    
    return {
        "keywords": [],
        "related_terms": [],
        "reason": "提取失敗",
        "time_sensitive": False
    }

async def extract_relevant_thread(conversation_context, query, grok3_api_key):
    """
    從對話歷史中提取相關帖子 ID，優先關鍵詞匹配，後備語義相似度分析。
    """
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "無對話歷史"
    
    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = query_keyword_result["keywords"] + query_keyword_result["related_terms"]
    
    follow_up_phrases = ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"]
    is_follow_up_query = any(phrase in query for phrase in follow_up_phrases)
    
    # 優先關鍵詞匹配
    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = title_keyword_result["keywords"] + title_keyword_result["related_terms"]
                common_keywords = set(query_keywords).intersection(set(title_keywords))
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords or is_follow_up_query:
                    return thread_id, title, message["content"], f"關鍵詞匹配：{common_keywords or query_keywords}, 追問詞：{is_follow_up_query}"
    
    # 後備語義相似度分析
    prompt = f"""
比較以下查詢和對話歷史，找出最相關的帖子 ID，基於語義相似度。
查詢：{query}
對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}
關鍵詞：{json.dumps(query_keywords, ensure_ascii=False)}
輸出格式：{{
  "thread_id": "帖子 ID",
  "title": "標題",
  "reason": "匹配原因"
}}
若無匹配，返回空對象 {{}}
"""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {grok3_api_key}"
    }
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "你是語義分析助手，專注於匹配 Reddit 和 LIHKG 帖子。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=CONFIG["parse_timeout"]
            ) as response:
                if response.status != 200:
                    logger.debug(f"相關帖子提取失敗：狀態碼={response.status}")
                    return None, None, None, "API 請求失敗"
                data = await response.json()
                result = json.loads(data["choices"][0]["message"]["content"])
                thread_id = result.get("thread_id")
                title = result.get("title")
                reason = result.get("reason", "未提供原因")
                if thread_id:
                    return thread_id, title, None, reason
                return None, None, None, "無匹配帖子"
    except Exception as e:
        logger.debug(f"相關帖子提取錯誤：{str(e)}")
        return None, None, None, f"提取錯誤：{str(e)}"

async def build_dynamic_prompt(query, conversation_context, metadata, thread_data, filters, intent, selected_source, grok3_api_key):
    # 確保 selected_source 是字典
    if isinstance(selected_source, str):
        source_name = selected_source
        source_type = "reddit" if "reddit" in source_name.lower() else "lihkg"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.warning(f"無效的 selected_source 類型：{type(selected_source)}，使用默認值")
        selected_source = {"source_name": "未知", "source_type": "lihkg"}
    
    parsed_query = await parse_query(query, conversation_context, grok3_api_key, selected_source.get("source_type", "lihkg"))
    intents = parsed_query["intents"]
    keywords = parsed_query["keywords"]
    related_terms = parsed_query["related_terms"]
    time_range = parsed_query["time_range"]
    
    system = (
        "你是社交媒體討論區（包括 LIHKG 和 Reddit）的數據助手，以繁體中文回答，"
        "語氣客觀輕鬆，專注於提供清晰且實用的資訊。引用帖子時使用 [帖子 ID: {thread_id}] 格式，"
        "禁止使用 [post_id: ...] 格式。根據用戶意圖動態選擇回應格式（例如列表、段落、表格等），"
        "確保結構清晰、內容連貫，且適合查詢的需求。"
    )
    
    context = (
        f"用戶問題：{query}\n"
        f"討論區：{selected_source.get('source_name', '未知')}\n"
        f"對話歷史：{json.dumps(conversation_context, ensure_ascii=False)}"
    )
    
    data = (
        f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n"
        f"帖子內容：{json.dumps(thread_data, ensure_ascii=False)}\n"
        f"篩選條件：{json.dumps(filters, ensure_ascii=False)}"
    )
    
    # 動態計算字數範圍
    word_min = 500
    word_max = 1500
    for intent_info in intents:
        intent = intent_info["intent"]
        min_w, max_w = CONFIG["default_word_ranges"].get(intent, (500, 1500))
        word_min = max(word_min, min_w)
        word_max = min(word_max, max_w)
    
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / CONFIG["max_prompt_length"], 1.0)
    context_factor = len(conversation_context) / 10
    keyword_factor = len(keywords) / CONFIG["max_keywords"]
    word_min = int(word_min + (word_max - word_min) * (length_factor * 0.3 + context_factor * 0.2 + keyword_factor * 0.1))
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.5 + context_factor * 0.1 + keyword_factor * 0.1))
    
    is_vague = len(keywords) < 2 and not any(kw in query for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"])
    
    # 動態生成指令並指定回應格式
    instruction_parts = []
    format_instructions = []
    source_type = selected_source.get("source_type", "lihkg")
    
    # 根據意圖數量動態調整指令
    intent_count = len(intents)
    if intent_count > 1:
        instruction_parts.append(
            f"綜合以下{intent_count}個意圖，生成一個連貫回應，聚焦關鍵詞 {keywords} 和相關詞 {related_terms}，"
            f"避免簡單合併各意圖的獨立段落，確保內容流暢、觀點融合，按主題組織討論，引用高點讚或最新回覆。"
        )
        format_instructions.append(
            "使用連貫段落格式，按主題分段，每段融合多個意圖的分析，引用 [帖子 ID: {thread_id}]，"
            "必要時使用表格或列表補充結構化信息（如情緒比例、排序結果）。"
        )
    
    for intent_info in intents[:4]:
        intent = intent_info["intent"]
        if intent == "summarize_posts":
            instruction_parts.append(f"總結最多5個帖子的討論內容，聚焦關鍵詞 {keywords}，簡明扼要。")
            format_instructions.append("融入段落，總結核心討論，引用 [帖子 ID: {thread_id}]。")
        elif intent == "analyze_sentiment":
            instruction_parts.append(f"分析最多5個帖子的情緒（正面、中立、負面），提供情緒比例，融入總結。")
            format_instructions.append("融入段落，總結情緒分析，必要時以表格列出 | 帖子 ID | 情緒 | 比例 |。")
        elif intent == "follow_up":
            instruction_parts.append(f"深入分析對話歷史中的帖子，聚焦問題關鍵詞 {keywords}，補充上下文。")
            format_instructions.append("融入段落，保持上下文連貫性，分段突出不同回覆觀點。")
        elif intent == "fetch_thread_by_id":
            instruction_parts.append(f"根據提供的帖子 ID 總結內容，突出核心討論。")
            format_instructions.append("融入段落，詳細總結指定帖子，引用 [帖子 ID: {thread_id}]。")
        elif intent == "general_query" and not is_vague:
            instruction_parts.append(f"提供與問題相關的簡化總結，基於元數據推測話題，保持簡潔。")
            format_instructions.append("融入段落，直接回答問題，必要時引用相關帖子。")
        elif intent == "list_titles":
            instruction_parts.append(f"列出最多15個帖子標題，聚焦關鍵詞 {keywords}，簡述相關性。")
            format_instructions.append("融入列表，每項包含 [帖子 ID: {thread_id}]、標題和相關性說明。")
        elif intent == "find_themed":
            instruction_parts.append(f"尋找與關鍵詞 {keywords} 和相關詞 {related_terms} 相關的帖子，突出主題關聯。")
            format_instructions.append("融入段落，強調主題相關性，引用 [帖子 ID: {thread_id}]。")
        elif intent == "fetch_dates":
            instruction_parts.append(f"提取最多5個帖子的發布或回覆日期，聚焦關鍵詞 {keywords}，融入總結。")
            format_instructions.append("融入表格，列出 | 帖子 ID | 標題 | 日期 |，後附簡短總結。")
        elif intent == "search_keywords":
            instruction_parts.append(f"搜索包含關鍵詞 {keywords} 和相關詞 {related_terms} 的帖子，強調關鍵詞匹配。")
            format_instructions.append("融入段落，突出關鍵詞匹配，引用 [帖子 ID: {thread_id}]。")
        elif intent == "recommend_threads":
            instruction_parts.append(f"推薦2-5個熱門或相關帖子，基於回覆數和點讚數，聚焦關鍵詞 {keywords}。")
            format_instructions.append("融入列表，每項包含 [帖子 ID: {thread_id}]、標題和推薦理由。")
        elif intent == "time_sensitive_analysis":
            instruction_parts.append(f"優先總結過去24小時的帖子，聚焦關鍵詞 {keywords}，突出時間敏感討論。")
            format_instructions.append("融入段落，標註回覆時間，引用 [帖子 ID: {thread_id}]。")
        elif intent == "contextual_analysis":
            instruction_parts.append(f"綜合最多5個帖子的討論，總結主要觀點，動態識別主題，分析情緒比例。")
            format_instructions.append("融入段落，按主題分段，結尾附情緒分佈表格 | 主題 | 比例 | 代表性帖子 |。")
        elif intent == "rank_topics":
            instruction_parts.append(f"按關注度排序最多5個帖子或話題，根據回覆數和點讚數，簡述熱度原因。")
            format_instructions.append("融入列表，每項包含 [帖子 ID: {thread_id}]、標題和熱度原因。")

    platform_instruction = (
        f"針對 {source_type} 平台，適當融入平台特定上下文（如 Reddit 的子版背景或 LIHKG 的討論區熱度）。"
    )
    
    combined_instruction = (
        f"根據以下意圖綜合生成一個連貫的回應（字數：{word_min}-{word_max}字）：{'；'.join(instruction_parts)}"
        f"選擇最適合的回應格式：簡介（1句）+按主題分段（每個主題融合多個意圖的分析）+{'；'.join(format_instructions)}"
        f"確保回應聚焦用戶問題，綜合所有意圖，內容不重複且流暢，引用帖子時標註 [帖子 ID: {{thread_id}}] {{標題}}。"
        f"優先考慮主要意圖（信心值最高者），其他意圖作為輔助，融合觀點避免機械性段落拼接。{platform_instruction}"
    )
    
    if time_range == "recent":
        combined_instruction += f"僅考慮過去24小時的帖子和回覆。"
    
    prompt = (
        f"[System]\n{system}\n"
        f"[Context]\n{context}\n"
        f"[Data]\n{data}\n"
        f"[Instructions]\n{combined_instruction}"
    )
    
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning("提示長度超過限制，縮減數據")
        thread_data = thread_data[:2]
        data = f"帖子元數據：{json.dumps(metadata, ensure_ascii=False)}\n篩選條件：{json.dumps(filters, ensure_ascii=False)}"
        prompt = (
            f"[System]\n{system}\n"
            f"[Context]\n{context}\n"
            f"[Data]\n{data}\n"
            f"[Instructions]\n{combined_instruction}"
        )
    
    logger.info(f"生成提示：查詢={query}, 提示長度={len(prompt)} 字符, intents={[i['intent'] for i in intents]}")
    return prompt

def format_context(conversation_context):
    """
    格式化對話歷史，簡化為結構化 JSON。
    """
    try:
        return [
            {"role": msg["role"], "content": msg["content"][:500]}
            for msg in conversation_context
        ]
    except Exception as e:
        logger.error(f"格式化對話歷史失敗：{str(e)}")
        return []

def extract_thread_metadata(metadata):
    """
    提取帖子元數據的關鍵字段。
    """
    try:
        return [
            {
                "thread_id": item["thread_id"],
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "like_count": item.get("like_count", 0),
                "last_reply_time": item.get("last_reply_time", 0)
            }
            for item in metadata
        ]
    except Exception as e:
        logger.error(f"提取元數據失敗：{str(e)}")
        return []

def instruction(list_name, value):
    """
    將值添加到指定的列表（instruction_parts 或 format_instructions）。
    """
    globals()[list_name].append(value)