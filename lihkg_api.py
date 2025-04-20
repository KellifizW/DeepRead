"""
LIHKG API 模組，負責從 LIHKG 論壇抓取帖子標題和回覆內容。
提供速率限制管理、錯誤處理和詳細日誌記錄。
主要函數：
- get_lihkg_topic_list：抓取指定分類的帖子標題。
- get_lihkg_thread_content：抓取指定帖子的回覆內容。
- get_category_name：返回分類名稱。
硬編碼參數（優化建議：移至配置文件）：
- max_replies=600
- per_page=25
- max_pages=3
- count=60
- items[:90]
- max_requests=30, period=60
- max_retries=3
- timeout=10
- sleep=0.5
"""

import aiohttp
import asyncio
import time
from datetime import datetime
import random
import hashlib
import logging

logger = logging.getLogger(__name__)

LIHKG_BASE_URL = "https://lihkg.com"
LIHKG_DEVICE_ID = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
LIHKG_COOKIE = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]

class RateLimiter:
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

rate_limiter = RateLimiter(max_requests=30, period=60)  # 修改：放寬至30次/60秒

def get_category_name(cat_id):
    categories = {
        "1": "吹水台", "2": "熱門台", "5": "時事台", "14": "上班台",
        "15": "財經台", "29": "成人台", "31": "創意台"
    }
    return categories.get(str(cat_id), "未知分類")

async def get_lihkg_topic_list(cat_id, start_page=1, max_pages=3, request_counter=0, last_reset=0, rate_limit_until=0):
    timestamp = int(time.time())
    url_template = f"{LIHKG_BASE_URL}/api_v2/thread/latest?cat_id={cat_id}&page={{page}}&count=60&type=now&order=now"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-REQUEST-TIME": str(timestamp),
        "Cookie": LIHKG_COOKIE,
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/category/{cat_id}",
    }
    
    items = []
    rate_limit_info = []
    max_retries = 3  # 硬編碼：建議移至配置文件
    current_time = time.time()
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        return {
            "items": items, "rate_limit_info": rate_limit_info, "request_counter": request_counter,
            "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    async with aiohttp.ClientSession() as session:
        for page in range(start_page, start_page + max_pages):
            if current_time - last_reset >= 60:
                request_counter = 0
                last_reset = current_time
            
            url = url_template.format(page=page)
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            headers["X-LI-DIGEST"] = digest
            
            for attempt in range(max_retries):
                try:
                    await rate_limiter.acquire()
                    request_counter += 1
                    logger.info(f"LIHKG API request: cat_id={cat_id}, page={page}, attempt={attempt+1}, url={url}")
                    async with session.get(url, headers=headers, timeout=10) as response:  # 硬編碼：timeout=10
                        if response.status == 429:
                            wait_time = int(response.headers.get("Retry-After", "5"))
                            rate_limit_until = time.time() + wait_time
                            rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                            logger.warning(f"Rate limit hit for cat_id={cat_id}, page={page}, waiting {wait_time:.2f} seconds")
                            await asyncio.sleep(wait_time)
                            continue
                        
                        if response.status != 200:
                            logger.error(f"Fetch failed: cat_id={cat_id}, page={page}, status={response.status}")
                            break
                        
                        data = await response.json()
                        logger.info(f"LIHKG API response: cat_id={cat_id}, page={page}, success={data.get('success', False)}")
                        if not data.get("success"):
                            logger.error(f"API error: cat_id={cat_id}, page={page}, message={data.get('error_message', 'Unknown')}")
                            break
                        
                        filtered_items = [item for item in data["response"]["items"] if item.get("title") and item.get("no_of_reply", 0) > 0]
                        items.extend(filtered_items)
                        logger.info(f"Fetched cat_id={cat_id}, page={page}, items={len(filtered_items)}")
                        break
                except Exception as e:
                    logger.error(f"Fetch error: cat_id={cat_id}, page={page}, attempt={attempt+1}, error={str(e)}")
                    break
            await asyncio.sleep(0.5)  # 修改：放寬至0.5秒
    return {
        "items": items[:90],  # 硬編碼：建議可配置
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": last_reset,
        "rate_limit_until": rate_limit_until
    }

async def fetch_thread_page(session, url, headers, thread_id, page, max_replies, rate_limiter, request_counter, rate_limit_until):
    rate_limit_info = []
    max_retries = 3  # 硬編碼：建議移至配置文件
    replies = []
    
    for attempt in range(max_retries):
        try:
            await rate_limiter.acquire()
            request_counter += 1
            logger.info(f"LIHKG API request: thread_id={thread_id}, page={page}, attempt={attempt+1}, url={url}")
            async with session.get(url, headers=headers, timeout=10) as response:  # 硬編碼：timeout=10
                if response.status == 429:
                    wait_time = int(response.headers.get("Retry-After", "5"))
                    rate_limit_until = time.time() + wait_time
                    rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                    logger.warning(f"Rate limit hit for thread_id={thread_id}, page={page}")
                    await asyncio.sleep(wait_time)
                    continue
                
                if response.status != 200:
                    logger.error(f"Fetch failed: thread_id={thread_id}, page={page}, status={response.status}")
                    break
                
                data = await response.json()
                logger.info(f"LIHKG API response: thread_id={thread_id}, page={page}, success={data.get('success', False)}")
                if not data.get("success"):
                    logger.warning(f"API error: thread_id={thread_id}, page={page}, message={data.get('error_message', 'Unknown')}")
                    break
                
                page_replies = data["response"].get("item_data", [])
                for reply in page_replies:
                    reply["like_count"] = int(reply.get("like_count", "0"))
                    reply["dislike_count"] = int(reply.get("dislike_count", "0"))
                    reply["reply_time"] = reply.get("reply_time", "0")
                
                remaining_slots = max_replies - len(replies)
                page_replies = page_replies[:remaining_slots]
                replies.extend(page_replies)
                logger.info(f"Fetched thread_id={thread_id}, page={page}, replies={len(page_replies)}")
                break
        except Exception as e:
            logger.error(f"Fetch error: thread_id={thread_id}, page={page}, attempt={attempt+1}, error={str(e)}")
            break
    
    return replies, page, request_counter, rate_limit_until, rate_limit_info

async def get_lihkg_thread_content(thread_id, cat_id=None, request_counter=0, last_reset=0, rate_limit_until=0, max_replies=600, fetch_last_pages=0, specific_pages=None, start_page=1):
    timestamp = int(time.time())
    url_template = f"{LIHKG_BASE_URL}/api_v2/thread/{thread_id}/page/{{page}}?order=reply_time"
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "X-LI-DEVICE": LIHKG_DEVICE_ID,
        "X-LI-REQUEST-TIME": str(timestamp),
        "Cookie": LIHKG_COOKIE,
        "Accept": "application/json",
        "Accept-Language": "zh-HK,zh-Hant;q=0.9,en;q=0.8",
        "Connection": "keep-alive",
        "Referer": f"{LIHKG_BASE_URL}/thread/{thread_id}",
    }
    
    replies = []
    fetched_pages = []
    thread_title = None
    total_replies = None
    total_pages = None
    rate_limit_info = []
    current_time = time.time()
    
    if current_time < rate_limit_until:
        rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit active until {datetime.fromtimestamp(rate_limit_until)}")
        logger.warning(f"Rate limit active for thread_id={thread_id}")
        return {
            "replies": replies, "title": thread_title, "total_replies": total_replies,
            "total_pages": total_pages, "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
            "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
        }
    
    async with aiohttp.ClientSession() as session:
        url = url_template.format(page=1)
        digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
        headers["X-LI-DIGEST"] = digest
        
        try:
            await rate_limiter.acquire()
            request_counter += 1
            logger.info(f"LIHKG API request: thread_id={thread_id}, page=1, url={url}")
            async with session.get(url, headers=headers, timeout=10) as response:
                if response.status == 429:
                    wait_time = int(response.headers.get("Retry-After", "5"))
                    rate_limit_until = time.time() + wait_time
                    rate_limit_info.append(f"{datetime.now():%Y-%m-%d %H:%M:%S} - Rate limit hit, waiting {wait_time:.2f} seconds")
                    logger.warning(f"Rate limit hit for thread_id={thread_id}, page=1")
                    return {
                        "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
                        "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
                    }
                
                if response.status != 200:
                    logger.error(f"Fetch failed: thread_id={thread_id}, page=1, status={response.status}")
                    return {
                        "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
                        "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
                    }
                
                data = await response.json()
                logger.info(f"LIHKG API response: thread_id={thread_id}, page=1, success={data.get('success', False)}")
                if not data.get("success"):
                    logger.warning(f"API error: thread_id={thread_id}, page=1, message={data.get('error_message', 'Unknown')}")
                    return {
                        "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
                        "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
                        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
                    }
                
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
                logger.info(f"Fetched thread_id={thread_id}, page=1, replies={len(page_replies)}")
        
        except Exception as e:
            logger.error(f"Fetch error: thread_id={thread_id}, page=1, error={str(e)}")
            return {
                "replies": [], "title": None, "total_replies": 0, "total_pages": 0,
                "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
                "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
            }
        
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
        
        for page in pages_to_fetch:
            if len(replies) >= max_replies:
                logger.info(f"Stopped fetching: thread_id={thread_id}, replies={len(replies)} reached max {max_replies}")
                break
            
            url = url_template.format(page=page)
            digest = hashlib.sha1(f"jeams$get${url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')}${timestamp}".encode()).hexdigest()
            headers["X-LI-DIGEST"] = digest
            
            page_replies, fetched_page, request_counter, rate_limit_until, page_rate_limit_info = await fetch_thread_page(
                session, url, headers, thread_id, page, max_replies, rate_limiter, request_counter, rate_limit_until
            )
            replies.extend(page_replies)
            fetched_pages.append(fetched_page)
            rate_limit_info.extend(page_rate_limit_info)
            await asyncio.sleep(0.5)  # 修改：放寬至0.5秒
    
    return {
        "replies": replies, "title": thread_title, "total_replies": total_replies,
        "total_pages": total_pages, "fetched_pages": fetched_pages, "rate_limit_info": rate_limit_info,
        "request_counter": request_counter, "last_reset": last_reset, "rate_limit_until": rate_limit_until
    }
