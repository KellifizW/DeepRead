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

# Centralized intent configuration
INTENT_CONFIG = {
    "summarize_posts": {
        "triggers": {
            "keywords": ["總結", "摘要", "概覽"],
            "confidence": 0.7,
            "reason": "Detected summarization query",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Summarize up to 5 threads, focusing on key points and keywords.",
        "prompt_format": "Paragraphs summarizing discussions, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "analyze_sentiment": {
        "triggers": {
            "keywords": ["情緒", "氣氛", "指數", "正負面", "sentiment", "mood"],
            "confidence": 0.90,
            "reason": "Detected sentiment analysis query",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Analyze sentiment of up to 5 threads (positive/neutral/negative).",
        "prompt_format": "Paragraphs with sentiment summary, optional table: | ID | Sentiment | Ratio |.",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 300,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "follow_up": {
        "triggers": {
            "keywords": ["詳情", "更多", "進一步", "點解", "為什麼", "原因", "講多D", "再講", "繼續", "仲有咩"],
            "confidence": 0.90,
            "reason": "Detected follow-up query",
        },
        "word_range": (1000, 4000),
        "prompt_instruction": "Deep dive into referenced threads, supplementing with context.",
        "prompt_format": "Paragraphs with cohesive follow-up, segmented by viewpoints.",
        "processing": {
            "post_limit": 2,
            "data_type": "replies",
            "max_replies": 400,
            "sort": "hot",
            "min_replies": 10,
            "sort_override": {"reddit": "new"},
        },
    },
    "fetch_thread_by_id": {
        "triggers": {
            "regex": r"(?:帖子\s*ID\s*[:=]?\s*|ID\s*[:=]?\s*)(\w+)",  # Matches "帖子 ID: 1kiu8i8" or "ID: 3926787"
            "confidence": 0.95,
            "reason": "Detected specific thread ID",
        },
        "word_range": (500, 2000),
        "prompt_instruction": "Summarize specified thread, highlighting core discussions.",
        "prompt_format": "Paragraphs summarizing thread, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 1,
            "data_type": "replies",
            "max_replies": 400,
            "sort": "hot",
            "min_replies": 0,
            "sort_override": {"reddit": "top"},
        },
    },
    "general_query": {
        "triggers": {
            "keywords": [],
            "confidence": 0.5,
            "reason": "Default general query",
        },
        "word_range": (500, 2000),
        "prompt_instruction": "Provide concise summary based on metadata, keeping it brief.",
        "prompt_format": "Paragraphs answering query, citing relevant threads if needed.",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 100,
            "sort": "hot",
            "min_replies": 10,
            "sort_override": {"reddit": "top"},
        },
    },
    "list_titles": {
        "triggers": {
            "keywords": ["列出", "標題", "清單", "所有標題"],
            "confidence": 0.95,
            "reason": "Detected list titles query",
        },
        "word_range": (500, 3000),
        "prompt_instruction": "List up to 15 thread titles, explaining relevance.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and relevance note.",
        "processing": {
            "post_limit": 15,
            "data_type": "metadata",
            "max_replies": 20,
            "sort": "hot",
            "min_replies": 5,
            "sort_override": {"reddit": "top"},
        },
    },
    "find_themed": {
        "triggers": {
            "keywords": ["主題", "類似", "相關"],
            "confidence": 0.85,
            "reason": "Detected themed query",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Find threads matching themes, highlighting thematic links.",
        "prompt_format": "Paragraphs emphasizing themes, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 20,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "fetch_dates": {
        "triggers": {
            "keywords": ["日期", "時間", "最近更新"],
            "confidence": 0.85,
            "reason": "Detected date-focused query",
        },
        "word_range": (500, 2000),
        "prompt_instruction": "Extract dates for up to 5 threads, integrating into summary.",
        "prompt_format": "Table: | ID | Title | Date |, followed by brief summary.",
        "processing": {
            "post_limit": 5,
            "data_type": "metadata",
            "max_replies": 100,
            "sort": "new",
            "min_replies": 5,
        },
    },
    "search_keywords": {
        "triggers": {
            "keywords": ["搜索", "查找", "關鍵詞"],
            "confidence": 0.85,
            "reason": "Detected keyword search query",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Search threads with matching keywords, emphasizing matches.",
        "prompt_format": "Paragraphs highlighting keyword matches, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 20,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "recommend_threads": {
        "triggers": {
            "keywords": ["推薦", "建議", "熱門"],
            "confidence": 0.85,
            "reason": "Detected recommendation query",
        },
        "word_range": (500, 3000),
        "prompt_instruction": "Recommend 2-5 threads based on replies/likes, justifying choices.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and recommendation reason.",
        "processing": {
            "post_limit": 5,
            "data_type": "metadata",
            "max_replies": 100,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "time_sensitive_analysis": {
        "triggers": {
            "keywords": ["今晚", "今日", "最近", "今個星期"],
            "confidence": 0.85,
            "reason": "Detected time-sensitive query",
        },
        "word_range": (500, 2000),
        "prompt_instruction": "Summarize threads from last 24 hours, focusing on recent discussions.",
        "prompt_format": "Paragraphs noting reply times, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 150,
            "sort": "new",
            "min_replies": 15,
        },
    },
    "contextual_analysis": {
        "triggers": {
            "keywords": ["Reddit", "LIHKG", "子版", "討論區"],
            "confidence": 0.85,
            "reason": "Detected platform-specific query",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Summarize 5 threads, identifying themes and sentiment ratios.",
        "prompt_format": "Paragraphs by theme, ending with table: | Theme | Ratio | Thread |.",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "rank_topics": {
        "triggers": {
            "keywords": ["熱門", "最多", "關注", "流行"],
            "confidence": 0.85,
            "reason": "Detected ranking query",
        },
        "word_range": (500, 2000),
        "prompt_instruction": "Rank up to 5 threads/topics by engagement, explaining reasons.",
        "prompt_format": "List with [帖子 ID: {thread_id}], title, and engagement reason.",
        "processing": {
            "post_limit": 5,
            "data_type": "metadata",
            "max_replies": 100,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "search_odd_posts": {
        "triggers": {
            "keywords": ["離奇", "怪事", "奇聞", "詭異", "不可思議", "怪談", "荒誕"],
            "confidence": 0.85,
            "reason": "Detected query for odd or unusual posts",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Summarize up to 5 threads with odd or unusual content, highlighting peculiar aspects.",
        "prompt_format": "Paragraphs summarizing odd discussions, citing [帖子 ID: {thread_id}].",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "min_replies": 15,
            "sort_override": {"reddit": "top"},
        },
    },
    "hypothetical_advice": {
        "triggers": {
            "keywords": ["你點睇", "你認為", "你會點", "你會如何", "如果你是", "假設你是", "你的看法", "你建議"],
            "regex": r"(你|Grok)\s*(點睇|認為|會\s*\w+|如何|是\s*.*\s*會|看法|建議)",
            "confidence": 0.90,
            "reason": "Detected hypothetical or advice-seeking query targeting Grok's perspective",
        },
        "word_range": (500, 2500),
        "prompt_instruction": "Answer as Grok, providing insights or advice for the hypothetical scenario or query, incorporating platform discussions if relevant.",
        "prompt_format": "Paragraphs with Grok's perspective, optionally citing [帖子 ID: {thread_id}] for context.",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 100,
            "sort": "hot",
            "min_replies": 10,
            "sort_override": {"reddit": "top"},
        },
    },
    "risk_warning": {
        "triggers": {
            "keywords": ["投資", "股票", "基金", "債券", "樓市", "房地產", "加密貨幣", "比特幣", "風險", "市場"],
            "confidence": 0.85,
            "reason": "Detected financial query requiring risk analysis",
        },
        "word_range": (700, 3000),
        "prompt_instruction": "Identify and analyze risk factors (e.g., policy changes, market volatility, negative news) in financial discussions, assessing their potential impact on the specified investment.",
        "prompt_format": "Table: | Risk Factor | Description | Potential Impact | Thread Reference |, followed by a summary paragraph.",
        "processing": {
            "post_limit": 5,
            "data_type": "both",
            "max_replies": 150,
            "sort": "hot",
            "sort_override": {"reddit": "controversial"},
            "min_replies": 15,
        },
    },
}

CONFIG = {
    "max_prompt_length": 240000,
    "max_parse_retries": 3,
    "parse_timeout": 90,
    "min_keywords": 1,
    "max_keywords": 8,
    "intent_confidence_threshold": 0.75,
}

async def call_grok3_api(
    payload: Dict,
    retries: int = CONFIG["max_parse_retries"],
    timeout: int = CONFIG["parse_timeout"],
    function_name: str = "unknown",
) -> Optional[Dict]:
    """Unified Grok 3 API call handler with retry and error logging."""
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {st.secrets['grok3key']}",
        }
    except KeyError:
        logger.error(f"API call failed in {function_name}: Missing Grok 3 API key")
        return None

    for attempt in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    GROK3_API_URL, headers=headers, json=payload, timeout=timeout
                ) as response:
                    if response.status != 200:
                        logger.warning(
                            f"API call failed in {function_name}: status={response.status}, attempt={attempt + 1}"
                        )
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
                        continue
                    data = await response.json()
                    if not data.get("choices"):
                        logger.warning(f"API call failed in {function_name}: no choices, attempt={attempt + 1}")
                        if attempt < retries - 1:
                            await asyncio.sleep(2)
                        continue
                    logger.info(
                        f"API call in {function_name}: status={response.status}, "
                        f"prompt_tokens={data.get('usage', {}).get('prompt_tokens', 0)}, "
                        f"completion_tokens={data.get('usage', {}).get('completion_tokens', 0)}"
                    )
                    return data
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.error(f"API call error in {function_name}: {str(e)}, attempt={attempt + 1}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
        except json.JSONDecodeError as e:
            logger.error(f"API response JSON decode error in {function_name}: {str(e)}, attempt={attempt + 1}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
    logger.warning(f"API call failed in {function_name} after {retries} attempts")
    return None

async def determine_post_limit(
    query: str, keywords: List[str], intents: List[Dict], source_type: str, grok3_api_key: str
) -> int:
    """Determine dynamic post limit (3-15) based on query specificity."""
    logger.info(f"Determining post_limit: query={query}, keywords={keywords}, intents={intents}")
    prompt = f"""
Analyze the query to determine the optimal number of posts (3-15) to fetch.
- Broad queries (e.g., '市場趨勢') require more posts (10-15).
- Specific queries (e.g., '某帖子詳情') require fewer posts (3-5).
- Consider keywords and intents for specificity.
Query: {query}
Keywords: {json.dumps(keywords, ensure_ascii=False)}
Intents: {json.dumps(intents, ensure_ascii=False)}
Source: {source_type}
Output Format: {{"post_limit": number, "reason": "reason (max 70 chars)"}}
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "You are an assistant determining optimal post limits."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    api_response = await call_grok3_api(payload, function_name="determine_post_limit")
    if api_response and api_response.get("choices"):
        response_content = api_response["choices"][0]["message"]["content"]
        logger.info(f"Raw API response in determine_post_limit: {response_content}")
        try:
            result = json.loads(response_content)
            post_limit = max(3, min(15, result.get("post_limit", 5)))
            logger.info(f"Dynamic post_limit: {post_limit}, reason={result.get('reason', 'No reason')}")
            return post_limit
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in determine_post_limit: {str(e)}, response_content={response_content}")
            # Attempt to fix truncated JSON
            if response_content.strip().endswith("...") or response_content.strip().endswith('"'):
                try:
                    fixed_content = response_content.rsplit(",", 1)[0] + "}"
                    result = json.loads(fixed_content)
                    post_limit = max(3, min(15, result.get("post_limit", 5)))
                    logger.info(f"Fixed truncated JSON: post_limit={post_limit}, reason={result.get('reason', 'Fixed JSON')}")
                    return post_limit
                except json.JSONDecodeError:
                    logger.warning("Failed to fix truncated JSON")
            # Fallback based on keyword count
            keyword_count = len(keywords)
            if keyword_count <= 2:
                post_limit = 3  # Specific query
            elif keyword_count <= 4:
                post_limit = 5  # Moderately specific
            else:
                post_limit = 10  # Broad query
            logger.info(f"Fallback post_limit: {post_limit}, reason=JSON parsing failed, keyword_count={keyword_count}")
            return post_limit
    # Fallback when API fails
    logger.warning("Failed to determine dynamic post_limit, using default")
    return 5

async def parse_query(
    query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg"
) -> Dict:
    """Parse user query to extract intents, keywords, thread IDs, and post_limit."""
    conversation_context = conversation_context or []
    if not isinstance(query, str):
        logger.error(f"Invalid query type: expected str, got {type(query)}")
        return {
            "intents": [{"intent": "summarize_posts", "confidence": 0.5, "reason": "Invalid query type"}],
            "keywords": [],
            "related_terms": [],
            "time_range": "all",
            "thread_ids": [],
            "reason": "Invalid query type",
            "confidence": 0.5,
            "post_limit": 5,
        }

    keyword_result = await extract_keywords(query, conversation_context, grok3_api_key, source_type)
    keywords = keyword_result.get("keywords", [])
    related_terms = keyword_result.get("related_terms", [])
    time_sensitive = keyword_result.get("time_sensitive", False)

    intents = []
    thread_ids = []
    query_lower = query.lower()
    reason = "Initial intent detection"

    # Step 1: Check for intent triggers (keywords or regex)
    for intent, config in INTENT_CONFIG.items():
        triggers = config.get("triggers", {})
        if "regex" in triggers:
            match = re.search(triggers["regex"], query, re.IGNORECASE)
            if match:
                if intent == "fetch_thread_by_id":
                    thread_id = match.group(1)
                    intents.append(
                        {
                            "intent": intent,
                            "confidence": triggers["confidence"],
                            "reason": f"{triggers['reason']}: {thread_id}",
                        }
                    )
                    thread_ids.append(thread_id)
                    logger.info(f"Intent triggered by regex: {intent}, thread_id: {thread_id}, match: {match.group(0)}")
                else:
                    intents.append(
                        {
                            "intent": intent,
                            "confidence": triggers["confidence"],
                            "reason": triggers["reason"],
                        }
                    )
                    logger.info(f"Intent triggered by regex: {intent}, match: {match.group(0)}")
        elif "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
            intents.append(
                {
                    "intent": intent,
                    "confidence": triggers["confidence"],
                    "reason": triggers["reason"],
                }
            )
            logger.info(f"Intent triggered by keywords: {intent}, keywords: {triggers['keywords']}")
            if intent == "time_sensitive_analysis":
                time_sensitive = True

    is_vague = len(keywords) < 2 and not any(
        kw in query_lower for kw in ["分析", "總結", "討論", "主題", "時事", "推薦", "熱門", "最多", "關注"]
    )
    has_multi_intent = any(ind in query_lower for ind in ["並且", "同時", "總結並", "列出並", "分析並"])

    # Step 2: Use API for semantic analysis if no clear intent or vague query
    if not intents or (is_vague and not has_multi_intent):
        logger.info("No clear intent or vague query, performing semantic analysis")
        prompt = f"""
Analyze the query to classify up to 4 intents, considering conversation history and keywords.
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Keywords: {json.dumps(keywords, ensure_ascii=False)}
Related Terms: {json.dumps(related_terms, ensure_ascii=False)}
Source: {source_type}
Is Vague: {is_vague}
Has Multi-Intent: {has_multi_intent}
Supported Intents: {json.dumps(list(INTENT_CONFIG.keys()), ensure_ascii=False)}
Output Format: {{"intents": [{{"intent": "intent", "confidence": 0.0-1.0, "reason": "reason"}}, ...], "reason": "overall reason"}}
Filter intents with confidence >= {CONFIG["intent_confidence_threshold"]}.
"""
        payload = {
            "model": "grok-3",
            "messages": [
                {"role": "system", "content": "You are a semantic analysis assistant, responding in Traditional Chinese."},
                *conversation_context,
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 300,
            "temperature": 0.5,
        }
        api_response = await call_grok3_api(payload, function_name="parse_query")
        if api_response and api_response.get("choices"):
            response_content = api_response["choices"][0]["message"]["content"]
            try:
                result = json.loads(response_content)
                intents = [
                    i for i in result.get("intents", []) if i["confidence"] >= CONFIG["intent_confidence_threshold"]
                ]
                reason = result.get("reason", "Semantic matching")
                logger.info(f"Semantic analysis result: intents={[i['intent'] for i in intents]}, reason={reason}")
            except json.JSONDecodeError as e:
                logger.error(f"Parse query response error: {str(e)}, response_content={response_content}")
                reason = f"JSON parsing error: {str(e)}"
                if response_content.strip().endswith("..."):
                    try:
                        fixed_content = response_content.rsplit(",", 1)[0] + "]}"
                        result = json.loads(fixed_content)
                        intents = [
                            i for i in result.get("intents", []) if i["confidence"] >= CONFIG["intent_confidence_threshold"]
                        ]
                        reason = result.get("reason", "Semantic matching (fixed truncated JSON)")
                        logger.info(f"Fixed truncated JSON in parse_query: {fixed_content[:100]}...")
                    except json.JSONDecodeError:
                        logger.warning("Failed to fix truncated JSON")
                        reason = f"Failed to fix truncated JSON: {str(e)}"
                inferred_intent = "summarize_posts"
                for intent, config in INTENT_CONFIG.items():
                    triggers = config.get("triggers", {})
                    if "keywords" in triggers and any(kw.lower() in query_lower for kw in triggers["keywords"]):
                        inferred_intent = intent
                        break
                intents = [
                    {
                        "intent": inferred_intent,
                        "confidence": 0.6,
                        "reason": f"JSON parsing failed, inferred from keywords: {keywords}",
                    }
                ]
        else:
            reason = "API call failed"
            intents = [
                {
                    "intent": "summarize_posts",
                    "confidence": 0.7,
                    "reason": "API call failed, default to summarization",
                }
            ]

    if not intents:
        reason = "No specific intent detected, default to summarization"
        intents = [
            {
                "intent": "summarize_posts",
                "confidence": 0.7,
                "reason": reason,
            }
        ]

    if is_vague and not has_multi_intent:
        intents = [max(intents, key=lambda x: x["confidence"])]

    # Step 3: Determine dynamic post_limit
    post_limit = await determine_post_limit(query, keywords, intents, source_type, grok3_api_key)

    time_range = "recent" if time_sensitive else "all"
    logger.info(
        f"Parsed query: intents={[i['intent'] for i in intents]}, keywords={keywords}, "
        f"thread_ids={thread_ids}, time_range={time_range}, post_limit={post_limit}, reason={reason}"
    )
    return {
        "intents": intents[:4],
        "keywords": keywords,
        "related_terms": related_terms,
        "time_range": time_range,
        "thread_ids": thread_ids,
        "reason": reason,
        "confidence": max(i["confidence"] for i in intents),
        "post_limit": post_limit,
    }

async def extract_keywords(
    query: str, conversation_context: List[Dict], grok3_api_key: str, source_type: str = "lihkg"
) -> Dict:
    """Extract semantic keywords and related terms from query."""
    generic_terms = ["post", "分享", "有咩", "什麼", "點樣", "如何", "係咩"]
    platform_terms = {
        "lihkg": ["吹水", "高登", "連登"],
        "reddit": ["YOLO", "DD", "subreddit"],
    }
    prompt = f"""
Extract {CONFIG['min_keywords']}-{CONFIG['max_keywords']} core keywords (prioritize nouns/verbs, exclude generic terms: {generic_terms}).
Generate up to 8 related terms (synonyms or platform terms: {platform_terms[source_type]}).
Set time_sensitive to true if query contains time-sensitive words (e.g., 今晚, 今日, 最近).
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Source: {source_type}
Output Format: {{"keywords": [], "related_terms": [], "reason": "logic (max 70 chars)", "time_sensitive": true/false}}
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": f"Semantic analysis assistant for {source_type} keywords."},
            *conversation_context,
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 200,
        "temperature": 0.3,
    }
    api_response = await call_grok3_api(payload, function_name="extract_keywords")
    if not api_response or not api_response.get("choices"):
        return {
            "keywords": [],
            "related_terms": [],
            "reason": "API call failed",
            "time_sensitive": False,
        }

    try:
        response_content = api_response["choices"][0]["message"]["content"]
        result = json.loads(response_content)
        keywords = [kw for kw in result.get("keywords", []) if kw.lower() not in generic_terms]
        logger.info(f"Extracted keywords: {result}")
        return {
            "keywords": keywords[:CONFIG["max_keywords"]],
            "related_terms": result.get("related_terms", [])[:8],
            "reason": result.get("reason", "No reason provided")[:70],
            "time_sensitive": result.get("time_sensitive", False),
        }
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Extract keywords response error: {str(e)}, response_content={response_content}")
        return {
            "keywords": [],
            "related_terms": [],
            "reason": f"Error: {str(e)}",
            "time_sensitive": False,
        }

async def extract_relevant_thread(
    conversation_context: List[Dict], query: str, grok3_api_key: str
) -> Tuple[Optional[str], Optional[str], Optional[str], str]:
    """Extract relevant thread ID from conversation history."""
    if not conversation_context or len(conversation_context) < 2:
        return None, None, None, "No conversation history"

    query_keyword_result = await extract_keywords(query, conversation_context, grok3_api_key)
    query_keywords = query_keyword_result["keywords"] + query_keyword_result["related_terms"]

    for message in reversed(conversation_context):
        if message["role"] == "assistant" and "帖子 ID" in message["content"]:
            matches = re.findall(r"\[帖子 ID: ([a-zA-Z0-9]+)\] ([^\n]+)", message["content"])
            for thread_id, title in matches:
                title_keyword_result = await extract_keywords(title, conversation_context, grok3_api_key)
                title_keywords = title_keyword_result["keywords"] + title_keyword_result["related_terms"]
                common_keywords = set(query_keywords).intersection(set(title_keywords))
                content_contains_keywords = any(kw.lower() in message["content"].lower() for kw in query_keywords)
                if common_keywords or content_contains_keywords:
                    return thread_id, title, message["content"], f"Keyword match: {common_keywords or query_keywords}"

    prompt = f"""
Compare query and conversation history to find the most relevant thread ID.
Query: {query}
Conversation History: {json.dumps(conversation_context, ensure_ascii=False)}
Keywords: {json.dumps(query_keywords, ensure_ascii=False)}
Output Format: {{"thread_id": "id", "title": "title", "reason": "reason"}}
Return empty object {{}} if no match.
"""
    payload = {
        "model": "grok-3",
        "messages": [
            {"role": "system", "content": "Semantic analysis assistant for thread matching."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 100,
        "temperature": 0.5,
    }
    api_response = await call_grok3_api(payload, function_name="extract_relevant_thread")
    if not api_response or not api_response.get("choices"):
        return None, None, None, "API call failed"

    try:
        response_content = api_response["choices"][0]["message"]["content"]
        result = json.loads(response_content)
        thread_id = result.get("thread_id")
        title = result.get("title")
        reason = result.get("reason", "No reason provided")
        return thread_id, title, None, reason if thread_id else "No matching thread"
    except (json.JSONDecodeError, KeyError) as e:
        logger.error(f"Extract relevant thread response error: {str(e)}, response_content={response_content}")
        return None, None, None, f"Error: {str(e)}"

async def build_dynamic_prompt(
    query: str,
    conversation_context: List[Dict],
    metadata: List[Dict],
    thread_data: List[Dict],
    filters: Dict,
    intent: str,
    selected_source: Dict,
    grok3_api_key: str,
) -> str:
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
        logger.warning(f"提示長度超過限制 {CONFIG['max_prompt_length']}，截斷帖子數據")
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
        logger.info(f"截斷後提示長度：{len(prompt)} 字符，保留帖子數={len(thread_data)}")

    logger.info(f"構建提示：查詢={query}, 長度={len(prompt)} 字符, 意圖={intent}")
    return prompt

def extract_thread_metadata(metadata: List[Dict]) -> List[Dict]:
    """Extract key fields from thread metadata."""
    try:
        return [
            {
                "thread_id": item["thread_id"],
                "title": item["title"],
                "no_of_reply": item.get("no_of_reply", 0),
                "like_count": item.get("like_count", 0),
                "last_reply_time": item.get("last_reply_time", 0),
            }
            for item in metadata
        ]
    except Exception as e:
        logger.error(f"提取元數據錯誤：{str(e)}")
        return []

def get_intent_processing_params(intent: str, source_type: str = "lihkg") -> Dict:
    """Retrieve processing parameters for a given intent, adjusting sort based on source_type."""
    params = INTENT_CONFIG.get(intent, INTENT_CONFIG["summarize_posts"]).get("processing", {}).copy()
    sort_override = params.get("sort_override", {})
    if sort_override and source_type.lower() in sort_override:
        params["sort"] = sort_override[source_type.lower()]
    # For LIHKG, replace 'confidence' with 'hot' as LIHKG does not support 'confidence'
    if source_type.lower() == "lihkg" and params["sort"] == "confidence":
        params["sort"] = "hot"
    return params
