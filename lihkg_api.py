"""
LIHKG API 模組，負責從 LIHKG 論壇抓取帖子標題和回覆內容。
提供速率限制管理、錯誤處理和日誌記錄功能。
主要函數：
- get_lihkg_topic_list：抓取指定分類的帖子標題。
- get_lihkg_thread_content：抓取指定帖子的回覆內容。
- get_category_name：返回分類名稱。
"""

import aiohttp
import asyncio
import time
from datetime import datetime
import random
import hashlib
import logging
import json

# 配置日誌記錄器
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 檔案處理器：寫入 app.log
file_handler = logging.FileHandler("app.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# 控制台處理器：輸出到 stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

# LIHKG API 基礎配置
LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"

# 用戶代理列表
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

class RateLimiter:
    """
    速率限制器，控制 API 請求頻率。
    """
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.requests = []

    async def acquire(self):
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.period]
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            logger.warning(f"Rate limit reached, waiting {wait_time:.2f} seconds")
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]
        self.requests.append(now)

class ApiClient:
    """
    LIHKG API 客戶端，處理共用請求邏輯。
    """
    def __init__(self, rate_limiter: RateLimiter):
        self.rate_limiter = rate_limiter
        self.base_url = LIHKG_BASE_URL
        self.device_id = LIHKG_DEVICE_ID
        self.cookie = LIHKG_COOKIE

    def generate_headers(self, url: str, timestamp: int):
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "X-LI-DEVICE": self.device_id,
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": digest,
            "Cookie": self.cookie,
            "Accept": "application/json",
            "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
            "Connection": "keep-alive",
            "Referer": url.rsplit('/', 1)[0]
        }

    async def get(self, url: str, function_name: str, params=None, max_retries=3, timeout=10):
        timestamp = int(time.time())
        headers = self.generate_headers(url, timestamp)
        rate_limit_info = []
        for attempt in range(max_retries):
            try:
                await self.rate_limiter.acquire()
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, params=params, timeout=timeout) as response:
                        status = "success" if response.status == 200 else f"failed_status_{response.status}"
                        logger.info(
                            json.dumps({
                                "event": "lihkg_api_request",
                                "function": function_name,
                                "url": url,
                                "status": status
                            }, ensure_ascii=False)
                        )
                        if response.status == 429:
                            wait_time = int(response.headers.get("Retry-After", "5"))
                            rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                            logger.warning(f"Rate limit hit for {function_name}, url={url}")
                            await asyncio.sleep(wait_time)
                            continue
                        if response.status != 200:
                            logger.error(f"Fetch failed: {function_name}, url={url}, status={response.status}")
                            break
                        data = await response.json()
                        if not data.get("success"):
                            logger.error(f"API error: {function_name}, url={url}, message={data.get('error_message', 'Unknown')}")
                            break
                        return data, rate_limit_info
            except Exception as e:
                logger.error(
                    json.dumps({
                        "event": "lihkg_api_request",
                        "function": function_name,
                        "url": url,
                        "status": "failed",
                        "error": str(e)
                    }, ensure_ascii=False)
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(2)
                    continue
                break
        return None, rate_limit_info

# 初始化速率限制器和客戶端
rate_limiter = RateLimiter(max_requests=50, period=60)
api_client = ApiClient(rate_limiter)

def get_category_name(cat_id):
    """
    根據分類 ID 返回分類名稱。
    """
    categories = {
        "1": "吹水台", "2": "熱門台", "5": "時事台", "14": "上班台",
        "15": "財經台", "29": "成人台", "31": "創意台"
    }
    return categories.get(str(cat_id), "未知分類")

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=3, request_counter=0, last_reset=0, rate_limit_until=0):
    """
    抓取指定分類的帖子標題列表。
    """
    items = []
    rate_limit_info = []
    current_time = time.time()
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        return {
            "items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter,
            "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    if current_time - last_reset >= 60:
        request_counter = 0
        last_reset = current_time
    
    for page in range(start_page, start_page + max_pages):
        url = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count=60&type=now&order=now"
        data, page_rate_limit_info = await api_client.get(url, "get_lihkg_topic_list")
        request_counter += 1
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response", {}).get("items"):
            filtered_items = [item for item in data["response"]["items"] if item.get("title") and item.get("no_of_reply", 0) > 0]
            items.extend(filtered_items)
            logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(filtered_items)}")
        else:
            logger.error(f"No data fetched for cat_id={cat_id}, page={page}")
        
        await asyncio.sleep(1)
    
    return {
        "items": items[:90],
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def get_lihkg_thread_content(thread_id, cat_id=None, request_counter=0, last_reset=0, rate_limit_until=0, max_replies=600, fetch_last_pages=0, specific_pages=None, start_page=1):
    """
    抓取指定帖子的回覆內容。
    """
    replies = []
    fetched_pages = []
    thread_title = None
    total_replies = None
    total_pages = None
    rate_limit_info = []
    current_time = time.time()
    max_replies = max(max_replies, 100)
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active for thread_id={thread_id}")
        return {
            "replies": replies, "title": thread_title, "total_replies": total_replies,
            "total_pages": total_pages, "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    if current_time - last_reset >= 60:
        request_counter = 0
        last_reset = current_time
    
    # 抓取第一頁
    url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/1?order=reply_time"
    data, page_rate_limit_info = await api_client.get(url, "get_lihkg_thread_content")
    request_counter += 1
    rate_limit_info.extend(page_rate_limit_info)
    
    if data and data.get("response"):
        response_data = data.get("response", {})
        thread_title = response_data.get("title") or response_data.get("thread", {}).get("title")
        total_replies = response_data.get("total_replies", 0)
        total_pages = response_data.get("total_page", 1)
        
        page_replies = response_data.get("item_data", [])
        for reply in page_replies:
            reply["like_count"] = int(reply.get("like_count", "0"))
            reply["dislike_count"] = int(reply.get("dislike_count", "0"))
            reply["reply_time"] = reply.get("reply_time", "0")
        replies.extend(page_replies[:max_replies])
        fetched_pages.append(1)
        logger.info(f"Fetched thread_id={thread_id}, page=1, replies={len(page_replies)}, total_stored={len(replies)}")
    else:
        logger.error(f"No data fetched for thread_id={thread_id}, page=1")
        return {
            "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
            "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    # 確定後續頁面
    pages_to_fetch = []
    if specific_pages:
        pages_to_fetch = [p for p in specific_pages if 1 <= p <= total_pages and p not in fetched_pages]
    elif fetch_last_pages > 0:
        start = max(start_page, 2)
        end = total_pages + 1
        if end - start < fetch_last_pages:
            start = max(2, end - fetch_last_pages)
        pages_to_fetch = list(range(start, end))[:fetch_last_pages]
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages and 1 <= p <= total_pages]
    elif start_page > 1:
        pages_to_fetch = list(range(start_page, total_pages + 1))
        pages_to_fetch = [p for p in pages_to_fetch if p not in fetched_pages]
    
    pages_to_fetch = sorted(set(pages_to_fetch))
    
    # 抓取後續頁面
    for page in pages_to_fetch:
        if len(replies) >= max_replies:
            logger.info(f"Stopped fetching: thread_id={thread_id}, replies={len(replies)} reached max {max_replies}")
            break
        
        url = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/{page}?order=reply_time"
        data, page_rate_limit_info = await api_client.get(url, "fetch_thread_page")
        request_counter += 1
        rate_limit_info.extend(page_rate_limit_info)
        
        if data and data.get("response"):
            page_replies = data["response"].get("item_data", [])
            for reply in page_replies:
                reply["like_count"] = int(reply.get("like_count", "0"))
                reply["dislike_count"] = int(reply.get("dislike_count", "0"))
                reply["reply_time"] = reply.get("reply_time", "0")
            remaining_slots = max_replies - len(replies)
            page_replies = page_replies[:remaining_slots]
            replies.extend(page_replies)
            fetched_pages.append(page)
            logger.info(f"Fetched thread_id={thread_id}, page={page}, replies={len(page_replies)}")
        await asyncio.sleep(1)
    
    return {
        "replies": replies, "title": thread_title, "total_replies": total_replies,
        "total_pages": total_pages, "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
    }
