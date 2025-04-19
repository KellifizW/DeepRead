import re
import time
import math
from datetime import datetime
import pytz

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

def clean_html(text: str) -> str:
    """清除HTML標籤"""
    clean = re.compile(r'<[^>]+>')
    return clean.sub('', text).strip()

def format_ts(timestamp: int) -> str:
    """格式化時間戳"""
    return datetime.fromtimestamp(timestamp, HONG_KONG_TZ).strftime('%Y-%m-%d %H:%M:%S')

def is_new_conversation(current: str, last: str) -> bool:
    """判斷是否新對話"""
    if not last:
        return True
    current_words = set(current.split())
    last_words = set(last.split())
    return len(current_words & last_words) < 2

class RateLimiter:
    """速率限制器"""
    def __init__(self, max_requests: int, period: float):
        self.max_requests = max_requests
        self.period = period
        self.requests = []
    
    async def acquire(self):
        """獲取請求許可"""
        now = time.time()
        self.requests = [t for t in self.requests if now - t < self.period]
        
        if len(self.requests) >= self.max_requests:
            wait_time = self.period - (now - self.requests[0])
            await asyncio.sleep(wait_time)
            self.requests = self.requests[1:]
            
        self.requests.append(now)