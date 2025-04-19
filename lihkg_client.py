import aiohttp
import asyncio
import time
import random
import hashlib
from typing import Dict, List, Optional

class LIHKGClient:
    def __init__(self):
        self.base_url = "https://lihkg.com"
        self.device_id = "5fa4ca23e72ee0965a983594476e8ad9208c808d"
        self.cookie = "PHPSESSID=ckdp63v3gapcpo8jfngun6t3av; __cfruid=019429f"
        self.rate_limiter = RateLimiter(20, 60)
        self.request_counter = 0
        self.last_reset = time.time()
        self.rate_limit_until = 0

    async def get_thread_list(self, cat_id: int, page: int = 1, count: int = 60) -> Dict:
        """獲取主題列表"""
        url = f"{self.base_url}/api_v2/thread/latest?cat_id={cat_id}&page={page}&count={count}"
        return await self._make_request(url)

    async def get_thread_content(
        self,
        thread_id: int,
        cat_id: Optional[int] = None,
        max_replies: int = 25,
        fetch_last_pages: int = 0
    ) -> Dict:
        """獲取主題內容"""
        url = f"{self.base_url}/api_v2/thread/{thread_id}/page/1"
        data = await self._make_request(url)
        
        if not data.get("success"):
            return {}
            
        response = data["response"]
        result = {
            "thread_id": thread_id,
            "title": response.get("title"),
            "total_replies": response.get("total_replies", 0),
            "item_data": response.get("item_data", []),
            "fetched_pages": [1]
        }
        
        # 如果需要抓取更多頁面
        if fetch_last_pages > 0:
            total_pages = (result["total_replies"] + 24) // 25
            pages_to_fetch = range(
                max(2, total_pages - fetch_last_pages + 1),
                total_pages + 1
            )
            
            for page in pages_to_fetch:
                page_data = await self._make_request(
                    f"{self.base_url}/api_v2/thread/{thread_id}/page/{page}"
                )
                if page_data.get("success"):
                    result["item_data"].extend(page_data["response"].get("item_data", []))
                    result["fetched_pages"].append(page)
                
                if len(result["item_data"]) >= max_replies:
                    break
        
        return result

    async def _make_request(self, url: str) -> Dict:
        """執行API請求"""
        await self._check_rate_limit()
        
        timestamp = int(time.time())
        headers = {
            "User-Agent": random.choice(USER_AGENTS),
            "X-LI-DEVICE": self.device_id,
            "X-LI-REQUEST-TIME": str(timestamp),
            "X-LI-DIGEST": self._generate_digest(url, timestamp),
            "Cookie": self.cookie,
            "Accept": "application/json"
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers, timeout=10) as resp:
                    if resp.status == 429:
                        await self._handle_rate_limit(resp)
                        return {"success": False, "error_message": "Rate limited"}
                    
                    data = await resp.json()
                    self.request_counter += 1
                    return data
                    
            except Exception as e:
                print(f"Request failed: {str(e)}")
                return {"success": False, "error_message": str(e)}

    def _generate_digest(self, url: str, timestamp: int) -> str:
        """生成LIHKG API需要的digest"""
        url_encoded = url.replace('[', '%5b').replace(']', '%5d').replace(',', '%2c')
        message = f"jeams$get${url_encoded}${timestamp}"
        return hashlib.sha1(message.encode()).hexdigest()

    async def _check_rate_limit(self):
        """檢查速率限制"""
        now = time.time()
        if now - self.last_reset >= 60:
            self.request_counter = 0
            self.last_reset = now
            
        if now < self.rate_limit_until:
            wait_time = self.rate_limit_until - now
            await asyncio.sleep(wait_time)

    async def _handle_rate_limit(self, response):
        """處理速率限制"""
        retry_after = int(response.headers.get("Retry-After", 5))
        self.rate_limit_until = time.time() + retry_after
        await asyncio.sleep(retry_after)

    def get_rate_limit_status(self) -> Dict:
        """獲取當前速率限制狀態"""
        return {
            "request_counter": self.request_counter,
            "last_reset": self.last_reset,
            "rate_limit_until": self.rate_limit_until
        }

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4_1 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4.1 Mobile/15E148 Safari/604.1"
]