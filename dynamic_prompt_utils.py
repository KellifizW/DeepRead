import aiohttp
import asyncio
import json
import logging
import re
from logging_config import configure_logger
import streamlit as st
from typing import Dict, List, Optional, Tuple

logger = configure_logger(__name__, "dynamic_prompt_utils.log")
GROK3_API_URL = "https://api.x.ai/v1/chat/completions"

# Default intent configuration
DEFAULT_INTENT_CONFIG = {
    "word_range": (500, 2000),
    "processing": {
        "post_limit": 5, "data_type": "both", "max_replies": 100,
        "sort": "hot", "min_replies": 10, "sort_override": {"reddit": "top"}
    }
}

# Simplified INTENT_CONFIG
INTENT_CONFIG = {
    "summarize_posts": {
        "triggers": {"keywords": ["總結", "摘要", "概覽"], "confidence": 0.7, "reason": "Summarization query"},
        "prompt_instruction": "Summarize up to 5 threads, focusing on key points.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "analyze_sentiment": {
        "triggers": {"keywords": ["情緒", "正負面"], "confidence": 0.9, "reason": "Sentiment analysis"},
        "prompt_instruction": "Analyze sentiment of up to 5 threads.",
        "prompt_format": "Paragraphs with table: | ID | Sentiment | Ratio |",
        "processing": {"max_replies": 300, "min_replies": 15},
        **DEFAULT_INTENT_CONFIG
    },
    "follow_up": {
        "triggers": {"keywords": ["詳情", "更多", "為什麼"], "confidence": 0.9, "reason": "Follow-up query"},
        "prompt_instruction": "Deep dive into threads, supplementing context.",
        "prompt_format": "Paragraphs by viewpoints",
        "word_range": (1000, 4000),
        "processing": {"post_limit": 2, "data_type": "replies", "max_replies": 400, "sort_override": {"reddit": "new"}},
        **DEFAULT_INTENT_CONFIG
    },
    "fetch_thread_by_id": {
        "triggers": {"regex": r"(?:帖子\s*ID\s*[:=]?\s*|ID\s*[:=]?\s*)(\w+)", "confidence": 0.95, "reason": "Specific thread ID"},
        "prompt_instruction": "Summarize specified thread.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {"post_limit": 1, "data_type": "replies", "max_replies": 400, "min_replies": 0},
        **DEFAULT_INTENT_CONFIG
    },
    "general_query": {
        "triggers": {"keywords": [], "confidence": 0.5, "reason": "Default query"},
        "prompt_instruction": "Provide concise summary based on metadata.",
        "prompt_format": "Paragraphs citing relevant threads.",
        **DEFAULT_INTENT_CONFIG
    },
    "list_titles": {
        "triggers": {"keywords": ["列出", "標題"], "confidence": 0.95, "reason": "List titles query"},
        "prompt_instruction": "List up to 15 thread titles with relevance.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, relevance.",
        "processing": {"post_limit": 15, "data_type": "metadata", "max_replies": 20, "min_replies": 5},
        **DEFAULT_INTENT_CONFIG
    },
    "find_themed": {
        "triggers": {"keywords": ["主題", "相關"], "confidence": 0.85, "reason": "Themed query"},
        "prompt_instruction": "Find threads matching themes.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {"post_limit": 20, "max_replies": 150, "min_replies": 15},
        **DEFAULT_INTENT_CONFIG
    },
    "fetch_dates": {
        "triggers": {"keywords": ["日期", "最近更新"], "confidence": 0.85, "reason": "Date-focused query"},
        "prompt_instruction": "Extract dates for up to 5 threads.",
        "prompt_format": "Table: | ID | Title | Date |, with summary.",
        "processing": {"post_limit": 5, "data_type": "metadata", "sort": "new", "min_replies": 5},
        **DEFAULT_INTENT_CONFIG
    },
    "search_keywords": {
        "triggers": {"keywords": ["搜索", "關鍵詞"], "confidence": 0.85, "reason": "Keyword search"},
        "prompt_instruction": "Search threads with matching keywords.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {"post_limit": 20, "max_replies": 150, "min_replies": 15},
        **DEFAULT_INTENT_CONFIG
    },
    "recommend_threads": {
        "triggers": {"keywords": ["推薦", "熱門"], "confidence": 0.85, "reason": "Recommendation query"},
        "prompt_instruction": "Recommend 2-5 threads based on engagement.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, reason.",
        **DEFAULT_INTENT_CONFIG
    },
    "time_sensitive_analysis": {
        "triggers": {"keywords": ["今晚", "今日", "最近"], "confidence": 0.85, "reason": "Time-sensitive query"},
        "prompt_instruction": "Summarize threads from last 24 hours.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        "processing": {"sort": "new"},
        **DEFAULT_INTENT_CONFIG
    },
    "contextual_analysis": {
        "triggers": {"keywords": ["Reddit", "LIHKG"], "confidence": 0.85, "reason": "Platform-specific query"},
        "prompt_instruction": "Summarize 5 threads with themes and sentiment.",
        "prompt_format": "Paragraphs with table: | Theme | Ratio | Thread |",
        **DEFAULT_INTENT_CONFIG
    },
    "rank_topics": {
        "triggers": {"keywords": ["熱門", "最多"], "confidence": 0.85, "reason": "Ranking query"},
        "prompt_instruction": "Rank up to 5 threads by engagement.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, reason.",
        **DEFAULT_INTENT_CONFIG
    },
    "search_odd_posts": {
        "triggers": {"keywords": ["離奇", "怪事"], "confidence": 0.85, "reason": "Odd posts query"},
        "prompt_instruction": "Summarize up to 5 threads with odd content.",
        "prompt_format": "Paragraphs citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "hypothetical_advice": {
        "triggers": {"keywords": ["你認為", "你建議"], "regex": r"(你|Grok)\s*(認為|建議)", "confidence": 0.9, "reason": "Advice-seeking query"},
        "prompt_instruction": "Answer as Grok with insights or advice.",
        "prompt_format": "Paragraphs, optionally citing [帖子 ID: {thread_id}].",
        **DEFAULT_INTENT_CONFIG
    },
    "risk_warning": {
        "triggers": {"keywords": ["投資", "股票", "風險"], "confidence": 0.85, "reason": "Financial risk query"},
        "prompt_instruction": "Analyze risk factors in financial discussions.",
        "prompt_format": "Table: | Risk Factor | Description | Impact | Thread |, with summary.",
        "processing": {"sort_override": {"reddit": "controversial"}},
        **DEFAULT_INTENT_CONFIG
    }
}

CONFIG = {
    "max_prompt_length": 240000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75
}

async def call_grok3_api(payload: Dict, function_name: str = "unknown", retries: int = CONFIG["max_parse_retries"], timeout: int = CONFIG["parse_timeout"]) -> Optional[Dict]:
    """Unified Grok 3 API call handler."""
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {st.secrets['grok3key']}"}
    except KeyError:
        logger.error(f"Missing Grok 3 API key in {function_name}")
        return None

    async with aiohttp.ClientSession() as session:
        for attempt in range(retries):
            try:
                async with session.post(GROK3_API_URL, headers=headers, json=payload, timeout=timeout) as response:
                    if response.status != 200 or not (data := await response.json()).get("choices"):
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
                        continue
                    logger.info(f"API call in {function_name}: status={response.status}, tokens={data.get('usage', {})}")
                    return data
            except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError) as e:
                if attempt < retries - 1:
                    await asyncio.sleep(2)
                else:
                    logger.warning(f"API call failed in {function_name} after {retries} attempts: {str(e)}")
    return None

async def determine_post_limit(query: str, keywords: List[str], intents: List[Dict], source_type: str, grok3_api_key: str) -> int:
    """Determine dynamic post limit (3-15) based on query specificity."""
    prompt = f"""
Analyze query for optimal post count (3-15).
Query: {query}
Keywords: {keywords}
Intents: {intents}
Source: {source_type}
Output: {{"post_limit": number, "reason": "reason (max 70 chars)"}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": "Post limit analyzer"}, {"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    if api_response := await call_grok3_api(payload, "determine_post_limit"):
        try:
            result = json.loads(api_response["choices"][0]["message"]["content"])
            return max(3, min(15, result.get("post_limit", 5)))
        except json.JSONDecodeError:
            keyword_count = len(keywords)
            return 3 if keyword_count <= 2 else 5 if keyword_count <= 4 else 10
    return 5

async def parse_query(query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg") -> Dict:
    """Parse query to extract intents, keywords, thread IDs, and post_limit."""
    if not isinstance(query, str):
        logger.error(f"Invalid query type: {type(query)}")
        return {"intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "Invalid query type"}], "keywords": [], "thread_ids": [], "reason": "Invalid query type", "confidence": 0.5, "post_limit": 5}

    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords, time_sensitive = keyword_result.get("keywords", []), keyword_result.get("time_sensitive", False)
    intents, thread_ids, query_lower, reason = [], [], query.lower(), "Initial intent detection"

    for intent, config in INTENT_CONFIG.items():
        triggers = config.get("triggers", {})
        if "regex" in triggers and (match := re.search(triggers["regex"], query, re.IGNORECASE)):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})
            if intent == "fetch_thread_by_id":
                thread_ids.append(match.group(1))
        elif "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
            intents.append({"intent": intent, "confidence": triggers["confidence"], "reason": triggers["reason"]})
            if intent == "time_sensitive_analysis":
                time_sensitive = True

    is_vague = len(keywords) < 2 and not any(kw in query_lower for kw in ["分析", "總結", "討論"])
    if not intents or is_vague:
        prompt = f"Analyze query for intents. Query: {query}, History: {conversation_context}, Source: {source_type}"
        payload = {"model": "grok-3", "messages": [{"role": "system", "content": "Semantic analysis assistant"}, *conversation_context, {"role": "user", "content": prompt}], "max_tokens": 300, "temperature": 0.5}
        if api_response := await call_grok3_api(payload, "parse_query"):
            try:
                result = json.loads(api_response["choices"][0]["message"]["content"])
                intents = [i for i in result.get("intents", []) if i["confidence"] >= CONFIG["intent_confidence_threshold"]]
                reason = result.get("reason", "Semantic matching")
            except json.JSONDecodeError:
                intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "JSON parsing failed"}]
        else:
            intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "API call failed"}]

    if not intents:
        intents = [{"intent": "summarize_posts", "confidence": 0.7, "reason": "Default to summarization"}]

    post_limit = await determine_post_limit(query, keywords, intents, source_type, grok3_api_key)
    logger.info(f"Parsed query: intents={[i['intent'] for i in intents]}, keywords={keywords}, post_limit={post_limit}")
    return {
        "intents": intents[:4], "keywords": keywords, "related_terms": keyword_result.get("related_terms", []),
        "time_range": "recent" if time_sensitive else "all", "thread_ids": thread_ids,
        "reason": reason, "confidence": max(i["confidence"] for i in intents), "post_limit": post_limit
    }

async def extract_keywords(query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg") -> Dict:
    """Extract semantic keywords and related terms."""
    generic_terms = ["post", "分享", "有咩", "什麼"]
    platform_terms = {"lihkg": ["吹水", "連登"], "reddit": ["subreddit"]}
    prompt = f"""
Extract {CONFIG['min_keywords']}-{CONFIG['max_keywords']} keywords (exclude: {generic_terms}).
Generate up to 8 related terms (e.g., {platform_terms[source_type]}).
Check for time-sensitive words (e.g., 今日, 最近).
Query: {query}
History: {conversation_context}
Source: {source_type}
Output: {{"keywords": [], "related_terms": [], "reason": "logic (max 70 chars)", "time_sensitive": true/false}}
"""
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": f"Keyword analyzer for {source_type}"}, *conversation_context, {"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0.3
    }
    if api_response := await call_grok3_api(payload, "extract_keywords"):
        try:
            result = json.loads(api_response["choices"][0]["message"]["content"])
            keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
            return {
                "keywords": keywords[:CONFIG["max_keywords"]],
                "related_terms": result.get("related_terms", [])[:8],
                "reason": result.get("reason", "Extracted keywords")[:70],
                "time_sensitive": result.get("time_sensitive", False)
            }
        except json.JSONDecodeError:
            pass
    return {"keywords": [], "related_terms": [], "reason": "API call failed", "time_sensitive": False}

async def extract_relevant_thread(conversation_context: List[Dict], query: str, grok3_api_key: str) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract relevant thread ID from history."""
    if len(conversation_context) < 2:
        return None, None, None, "No conversation history"

    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = query_keyword_result["keywords"] + query_keyword_result["related_terms"]

    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = title_keyword_result["keywords"] + title_keyword_result["related_terms"]
                common_keywords = set(query_keywords).intersection(title_keywords)
                if common_keywords or any(kw.lower() in message["content"].lower() for kw in query_keywords):
                    return thread_id, title, message["content"], f"Keyword match: {common_keywords or query_keywords}"

    prompt = f"Find relevant thread ID. Query: {query}, History: {conversation_context}, Keywords: {query_keywords}"
    payload = {
        "model": "grok-3",
        "messages": [{"role": "system", "content": "Thread matching assistant"}, {"role": "user", "content": prompt}],
        "max_tokens": 100,
        "temperature": 0.5
    }
    if api_response := await call_grok3_api(payload, "extract_relevant_thread"):
        try:
            result = json.loads(api_response["choices"][0]["message"]["content"])
            thread_id = result.get("thread_id")
            return thread_id, result.get("title"), None, result.get("reason", "No reason") if thread_id else "No match"
        except json.JSONDecodeError:
            pass
    return None, None, None, "API call failed"

async def build_dynamic_prompt(query: str, conversation_context: List[Dict], metadata: List[Dict], thread_data: List[Dict], filters: Dict, intent: str, selected_source: Dict, grok3_api_key: str) -> str:
    """Build dynamic prompt based on intent and data."""
    if isinstance(selected_source, str):
        source_name = selected_source
        source_type = "reddit" if "reddit" in source_name.lower() else "lihkg"
        selected_source = {"source_name": source_name, "source_type": source_type}
    elif not isinstance(selected_source, dict):
        logger.warning(f"Invalid selected_source type: {type(selected_source)}, using default")
        selected_source = {"source_name": "Unknown", "source_type": "lihkg"}

    source_name = selected_source.get("source_name", "Unknown")
    source_type = selected_source.get("source_type", "lihkg")

    system = (
        f"You are a data assistant for {source_name} ({source_type}), responding in Traditional Chinese "
        "with a clear, concise, and engaging tone. Cite threads using [帖子 ID: {{thread_id}}] format. "
        "Dynamically choose response format (e.g., paragraphs, lists, tables) based on intent."
    )
    context = (
        f"Query: {query}\n"
        f"Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}\n"
        f"Platform: {source_name}"
    )
    # Inline extract_thread_metadata
    metadata = [{"thread_id": item["thread_id"], "title": item["title"], "no_of_reply": item.get("no_of_reply", 0), "like_count": item.get("like_count", 0), "last_reply_time": item.get("last_reply_time", 0)} for item in metadata]
    data = (
        f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
        f"Thread Data: {json.dumps(thread_data, ensure_ascii=False)}\n"
        f"Filters: {json.dumps(filters, ensure_ascii=False)}"
    )

    intent_config = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"])
    word_min, word_max = intent_config["word_range"]
    prompt_length = len(context) + len(data) + len(system) + 500
    length_factor = min(prompt_length / (CONFIG["max_prompt_length"] * 0.8), 1.0)
    word_min = int(word_min + (word_max - word_min) * length_factor * 0.7)
    word_max = int(word_min + (word_max - word_min) * (1 + length_factor * 0.7))

    instruction_parts = [intent_config["prompt_instruction"]]
    format_instructions = [intent_config["prompt_format"]]

    platform_instruction = (
        f"For {source_type}, incorporate platform-specific context (e.g., Reddit subreddits, LIHKG trends)."
    )
    combined_instruction = (
        f"Generate a cohesive response ({word_min}-{word_max} words): {'; '.join(instruction_parts)} "
        f"Use format: Intro (1 sentence) + thematic paragraphs + {'; '.join(format_instructions)} "
        f"Focus on query, ensure fluency, cite [帖子 ID: {{thread_id}}] {{title}}. {platform_instruction}"
    )

    prompt = f"[System]\n{system}\n[Context]\n{context}\n[Data]\n{data}\n[Instructions]\n{combined_instruction}"
    if len(prompt) > CONFIG["max_prompt_length"]:
        logger.warning(f"Prompt exceeds {CONFIG['max_prompt_length']}, truncating thread data")
        thread_data = [
            {
                **data,
                "replies": data.get("replies", [])[:10],
                "total_fetched_replies": min(len(data.get("replies", [])), 10)
            }
            for data in thread_data[:3]
        ]
        data = (
            f"Metadata: {json.dumps(metadata, ensure_ascii=False)}\n"
            f"Thread Data: {json.dumps(thread_data, ensure_ascii=False)}\n"
            f"Filters: {json.dumps(filters, ensure_ascii=False)}"
        )
        prompt = f"[System]\n{system}\n[Context]\n{context}\n[Data]\n{data}\n[Instructions]\n{combined_instruction}"
        logger.info(f"Truncated prompt: length={len(prompt)}, threads={len(thread_data)}")

    logger.info(f"Built prompt: query={query}, length={len(prompt)}, intent={intent}")
    return prompt
