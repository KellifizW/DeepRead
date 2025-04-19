import asyncio
import time
import math
from datetime import datetime
from typing import Dict, List
from utils import clean_html, format_ts

class DataProcessor:
    def __init__(self, lihkg_client, grok_client):
        self.lihkg = lihkg_client
        self.grok = grok_client
        self.cache_duration = 600  # 10分鐘快取

    async def process_user_question(self, user_query: str, cat_id: int, analysis: Dict) -> Dict:
        """處理用戶問題完整流程"""
        # 階段1：獲取初始帖子列表
        initial_threads = await self._fetch_initial_threads(cat_id, analysis)
        
        # 階段2：本地篩選
        filtered_threads = self._local_filter(initial_threads, analysis.get("filters", {}))
        
        # 階段3：Grok3標題篩選
        screening = await self.grok.screen_threads(user_query, filtered_threads)
        top_thread_ids = screening.get("top_thread_ids", [])
        
        # 階段4：獲取候選帖子內容
        candidate_data = await self._fetch_candidate_content(top_thread_ids, cat_id, analysis)
        
        # 階段5：最終處理
        final_data = await self._process_final_threads(
            candidate_data, 
            cat_id, 
            analysis.get("reply_limit", 75)
        )
        
        return {
            "metadata": self._prepare_metadata(final_data),
            "content": final_data,
            "rate_limit": self.lihkg.get_rate_limit_status()
        }

    async def _fetch_initial_threads(self, cat_id: int, analysis: Dict) -> List[Dict]:
        """獲取初始帖子列表"""
        threads = []
        for page in range(1, 4):  # 最多3頁
            result = await self.lihkg.get_thread_list(
                cat_id=cat_id,
                page=page,
                count=60
            )
            threads.extend(result.get("items", []))
            if len(threads) >= 90:  # 最多90個
                break
        return threads

    def _local_filter(self, threads: List[Dict], filters: Dict) -> List[Dict]:
        """本地篩選條件"""
        min_replies = filters.get("min_replies", 20)
        min_likes = filters.get("min_likes", 10)
        recent_only = filters.get("recent_only", False)
        
        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        today_ts = int(today.timestamp())
        
        return [
            t for t in threads
            if t.get("no_of_reply", 0) >= min_replies
            and t.get("like_count", 0) >= min_likes
            and (not recent_only or int(t.get("last_reply_time", 0)) >= today_ts)
        ]

    async def _fetch_candidate_content(self, thread_ids: List[int], cat_id: int, analysis: Dict) -> Dict:
        """獲取候選帖子內容"""
        data = {}
        for thread_id in thread_ids[:analysis.get("post_limit", 3)]:
            content = await self.lihkg.get_thread_content(
                thread_id=thread_id,
                cat_id=cat_id,
                max_replies=25
            )
            if content:
                data[str(thread_id)] = self._process_thread_content(content)
        return data

    def _process_thread_content(self, raw_data: Dict) -> Dict:
        """處理原始帖子內容"""
        return {
            "thread_id": raw_data.get("thread_id"),
            "title": raw_data.get("title", "無標題"),
            "replies": [
                {
                    "msg": clean_html(reply.get("msg", "")),
                    "like_count": reply.get("like_count", 0),
                    "dislike_count": reply.get("dislike_count", 0),
                    "reply_time": reply.get("reply_time", 0)
                }
                for reply in raw_data.get("item_data", [])
            ],
            "total_replies": raw_data.get("total_replies", 0),
            "fetched_pages": [1]  # 默認已抓取第一頁
        }

    async def _process_final_threads(self, threads: Dict, cat_id: int, reply_limit: int) -> Dict:
        """處理最終選定的帖子"""
        results = {}
        for thread_id, data in threads.items():
            # 計算需要抓取的頁數 (60%規則)
            total_pages = math.ceil(data["total_replies"] / 25)
            target_pages = math.ceil(total_pages * 0.6)
            remaining_pages = max(0, target_pages - len(data["fetched_pages"]))
            
            if remaining_pages > 0:
                # 抓取剩餘頁面
                additional_data = await self.lihkg.get_thread_content(
                    thread_id=thread_id,
                    cat_id=cat_id,
                    max_replies=reply_limit,
                    fetch_last_pages=remaining_pages
                )
                if additional_data:
                    data["replies"].extend([
                        {
                            "msg": clean_html(reply.get("msg", "")),
                            "like_count": reply.get("like_count", 0),
                            "dislike_count": reply.get("dislike_count", 0),
                            "reply_time": reply.get("reply_time", 0)
                        }
                        for reply in additional_data.get("item_data", [])
                    ])
                    data["fetched_pages"].extend(additional_data.get("fetched_pages", []))
            
            # 按點讚數排序
            data["replies"] = sorted(
                data["replies"],
                key=lambda x: x["like_count"],
                reverse=True
            )[:reply_limit]
            
            results[thread_id] = data
        return results

    def _prepare_metadata(self, threads: Dict) -> List[Dict]:
        """準備Grok3分析用的元數據"""
        return [
            {
                "thread_id": data["thread_id"],
                "title": data["title"],
                "no_of_reply": data["total_replies"],
                "like_count": max((r["like_count"] for r in data["replies"]), default=0),
                "dislike_count": max((r["dislike_count"] for r in data["replies"]), default=0)
            }
            for data in threads.values()
        ]