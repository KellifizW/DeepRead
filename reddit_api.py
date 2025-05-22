import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from contextlib import asynccontextmanager
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "reddit_api.log")

class RedditClientManager:
    def __init__(self):
        try:
            credentials = st.secrets["reddit"]["credentials"]
            if not credentials or len(credentials) < 1:
                raise KeyError("Reddit credentials are missing or empty")
            self.clients = [None] * len(credentials)
            self.request_counters = [0] * len(credentials)
            self.last_resets = [time.time()] * len(credentials)
            self.rate_limit_requests_per_minute = 60  # Reddit API: ~600 次/10 分鐘
            self.current_client_index = 0
            self.cache = {"topic": {}, "thread": {}}
            self.cache_durations = {"topic": 1800, "thread": 600}
            self.max_cache_size = 100
        except KeyError as e:
            logger.error(f"Secrets 配置錯誤：{str(e)}")
            raise

    @asynccontextmanager
    async def get_client(self):
        for i in range(len(self.clients)):
            client_idx = (self.current_client_index + i) % len(self.clients)
            if self.clients[client_idx] is None:
                self.clients[client_idx] = await self._init_reddit_client(client_idx)
            try:
                self.request_counters[client_idx], self.last_resets[client_idx] = await self._handle_rate_limit(client_idx)
                self.current_client_index = client_idx
                yield self.clients[client_idx]
                return
            except Exception as e:
                logger.error(f"客戶端 {client_idx} 獲取失敗：{str(e)}", exc_info=True)
        raise Exception("所有 Reddit 客戶端均無法使用")

    async def _init_reddit_client(self, client_idx):
        try:
            cred = st.secrets["reddit"]["credentials"][client_idx]
            user_agent = f"python:LIHKGChatBot:v1.0.{client_idx} (by /u/{st.secrets['reddit']['username']})"
            reddit = asyncpraw.Reddit(
                client_id=cred["client_id"],
                client_secret=cred["client_secret"],
                username=st.secrets["reddit"]["username"],
                password=st.secrets["reddit"]["password"],
                user_agent=user_agent,
                timeout=15
            )
            user = await reddit.user.me()
            logger.info(f"Reddit 客戶端 {client_idx} 初始化成功，已認證用戶：{user.name}")
            response = await reddit.get("api/v1/me")
            headers = response.get("headers", {}) if isinstance(response, dict) else getattr(response, 'headers', {})
            logger.info(
                f"客戶端 {client_idx} 速率限制資訊："
                f"X-Ratelimit-Remaining={headers.get('X-Ratelimit-Remaining', '未知')}, "
                f"X-Ratelimit-Reset={headers.get('X-Ratelimit-Reset', '未知')}"
            )
            return reddit
        except Exception as e:
            logger.error(f"Reddit 客戶端 {client_idx} 初始化失敗：{str(e)}", exc_info=True)
            raise

    async def _handle_rate_limit(self, client_idx):
        current_time = time.time()
        if self.request_counters[client_idx] >= self.rate_limit_requests_per_minute:
            wait_time = 60 - (current_time - self.last_resets[client_idx])
            if wait_time > 0:
                logger.warning(f"客戶端 {client_idx} 達到速率限制，切換客戶端")
                raise Exception("速率限制，嘗試下一個客戶端")
            self.request_counters[client_idx] = 0
            self.last_resets[client_idx] = current_time
        self.request_counters[client_idx] += 1
        return self.request_counters[client_idx], self.last_resets[client_idx]

    def clean_cache(self, cache_type):
        cache = self.cache[cache_type]
        duration = self.cache_durations[cache_type]
        current_time = time.time()
        expired_keys = [k for k, v in cache.items() if current_time - v["timestamp"] > duration]
        for k in expired_keys:
            del cache[k]
        if len(cache) > self.max_cache_size:
            sorted_keys = sorted(cache.items(), key=lambda x: x[1]["timestamp"])[:len(cache) - self.max_cache_size]
            for k, _ in sorted_keys:
                del cache[k]
            logger.info(f"清理 {cache_type} 緩存，移除 {len(sorted_keys)} 個條目，當前大小：{len(cache)}")

client_manager = RedditClientManager()

async def _handle_api_error(e, context, rate_limit_info):
    if isinstance(e, asyncprawcore.exceptions.RequestException) and "Session is closed" in str(e):
        logger.error(f"{context} 失敗：Session is closed", exc_info=True)
        rate_limit_info.append({"message": f"{context} 失敗：Session is closed"})
    elif isinstance(e, asyncio.TimeoutError):
        logger.error(f"{context} 超時", exc_info=True)
        rate_limit_info.append({"message": f"{context} 超時"})
    elif isinstance(e, asyncpraw.exceptions.RedditAPIException):
        logger.error(f"{context} 失敗：{str(e)}, 回應頭部：{getattr(e, 'response', {}).get('headers', '無')}", exc_info=True)
        rate_limit_info.append({"message": f"{context} 失敗：{str(e)}"})
    else:
        logger.error(f"{context} 失敗：{str(e)}", exc_info=True)
        rate_limit_info.append({"message": f"{context} 失敗：{str(e)}"})

async def format_submission(submission):
    hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
    return {
        "thread_id": submission.id,
        "title": submission.title,
        "no_of_reply": submission.num_comments,
        "last_reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S"),
        "like_count": submission.score
    }

async def get_reddit_topic_list(subreddit, start_page=1, max_pages=1, sort="confidence"):
    cache_key = f"{subreddit}_{start_page}_{max_pages}_{sort}"
    client_manager.clean_cache("topic")
    
    if cache_key in client_manager.cache["topic"]:
        cached_data = client_manager.cache["topic"][cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.cache_durations["topic"]:
            logger.info(f"使用緩存數據，子版：{subreddit}, 鍵：{cache_key}")
            return cached_data["data"]
    
    items = []
    rate_limit_info = []
    total_limit = 250
    items_per_page = 100
    start_index = (start_page - 1) * items_per_page
    end_index = start_index + (max_pages * items_per_page)
    
    sort_methods = {
        "confidence": lambda obj: obj.hot(limit=total_limit),
        "new": lambda obj: obj.new(limit=total_limit),
        "top": lambda obj: obj.top(time_filter="day", limit=total_limit),
        "controversial": lambda obj: obj.controversial(time_filter="day", limit=total_limit),
        "rising": lambda obj: obj.rising(limit=total_limit)
    }
    sort = sort if sort in sort_methods else "confidence"
    
    for client_idx in range(len(client_manager.clients)):
        async with client_manager.get_client() as reddit:
            for attempt in range(3):
                try:
                    subreddit_obj = await reddit.subreddit(subreddit)
                    logger.info(f"開始抓取子版 {subreddit}，排序：{sort}，客戶端：{client_idx}，嘗試：{attempt + 1}")
                    async with asyncio.timeout(60):
                        submissions = [s async for s in sort_methods[sort](subreddit_obj)]
                    for idx, submission in enumerate(submissions):
                        if idx < start_index or idx >= end_index:
                            continue
                        try:
                            async with asyncio.timeout(10):
                                items.append(await format_submission(submission))
                        except Exception as e:
                            _handle_api_error(e, f"處理貼文 {getattr(submission, 'id', 'unknown')}", rate_limit_info)
                    logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
                    result = {
                        "items": items,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": client_manager.request_counters[client_idx],
                        "last_reset": client_manager.last_resets[client_idx],
                        "rate_limit_until": 0,
                        "total_posts": len(items)
                    }
                    client_manager.cache["topic"][cache_key] = {"timestamp": time.time(), "data": result}
                    return result
                except Exception as e:
                    _handle_api_error(e, f"抓取子版 {subreddit}", rate_limit_info)
                    if attempt < 2 and "Session is closed" in str(e):
                        await asyncio.sleep(2)
                        continue
                    break
    logger.warning(f"所有客戶端無法抓取子版 {subreddit}")
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
        "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
        "rate_limit_until": 0,
        "total_posts": len(items)
    }

async def collect_comments(submission, max_comments, sort="confidence"):
    comments = []
    rate_limit_info = []
    
    try:
        submission.comment_sort = sort
        submission._comments = None
        await submission.load()
        
        for attempt in range(3):
            try:
                async with asyncio.timeout(60):
                    logger.info(f"展開 MoreComments，貼文ID={submission.id}，嘗試={attempt + 1}")
                    start_time = time.time()
                    await submission.comments.replace_more(limit=100)
                    logger.info(f"展開 MoreComments 完成，耗時={time.time() - start_time:.2f} 秒")
                    break
            except Exception as e:
                _handle_api_error(e, f"展開 MoreComments 貼文 {submission.id}", rate_limit_info)
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return comments[:max_comments], rate_limit_info
        
        async for comment in submission.comments:
            try:
                async with asyncio.timeout(10):
                    if hasattr(comment, 'body') and comment.body:
                        hk_time = datetime.fromtimestamp(comment.created_utc, tz=HONG_KONG_TZ)
                        comments.append({
                            "reply_id": comment.id,
                            "msg": comment.body,
                            "like_count": comment.score,
                            "reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S")
                        })
                        if len(comments) >= max_comments:
                            break
            except Exception as e:
                _handle_api_error(e, f"處理評論 {getattr(comment, 'id', 'unknown')}", rate_limit_info)
        return comments[:max_comments], rate_limit_info
    except Exception as e:
        _handle_api_error(e, f"收集評論 貼文 {submission.id}", rate_limit_info)
        return comments[:max_comments], rate_limit_info

async def get_reddit_thread_content(post_id, subreddit, max_comments=100, sort="confidence"):
    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}_{sort}"
    client_manager.clean_cache("thread")
    
    if cache_key in client_manager.cache["thread"]:
        cached_data = client_manager.cache["thread"][cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.cache_durations["thread"]:
            logger.info(f"使用緩存數據，貼文：{post_id}")
            return cached_data["data"]
    
    rate_limit_info = []
    async with client_manager.get_client() as reddit:
        try:
            request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
            async with asyncio.timeout(10):
                submission = await reddit.submission(id=post_id)
            if not submission:
                logger.error(f"無法獲取貼文：{post_id}")
                rate_limit_info.append({"message": f"無法獲取貼文 {post_id}"})
                return {
                    "title": "",
                    "total_replies": 0,
                    "last_reply_time": "1970-01-01 00:00:00",
                    "like_count": 0,
                    "replies": [],
                    "fetched_pages": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": 0
                }
            
            comments, comments_rate_limit_info = await collect_comments(submission, max_comments, sort)
            rate_limit_info.extend(comments_rate_limit_info)
            request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
            
            result = {
                "title": submission.title,
                "total_replies": submission.num_comments,
                "last_reply_time": datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                "like_count": submission.score,
                "replies": comments,
                "fetched_pages": [1],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": last_reset,
                "rate_limit_until": 0,
                "replies_count": len(comments)
            }
            client_manager.cache["thread"][cache_key] = {"timestamp": time.time(), "data": result}
            logger.info(f"抓取貼文 {post_id} 完成，回覆數：{len(comments)}")
            return result
        except Exception as e:
            _handle_api_error(e, f"抓取貼文 {post_id}", rate_limit_info)
            return {
                "title": "",
                "total_replies": 0,
                "last_reply_time": "1970-01-01 00:00:00",
                "like_count": 0,
                "replies": [],
                "fetched_pages": [],
                "rate_limit_info": rate_limit_info,
                "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                "rate_limit_until": 0
            }

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=100, sort="confidence"):
    results = []
    rate_limit_info = []
    total_posts = 0
    total_replies = 0
    
    async with client_manager.get_client() as reddit:
        try:
            logger.info(f"開始批次抓取貼文：{post_ids}")
            ids = [f"t3_{pid}" for pid in post_ids if pid and pid.strip() and pid.lower() != "id"]
            if not ids:
                logger.error("無有效貼文 ID")
                rate_limit_info.append({"message": "無有效貼文 ID"})
                return {
                    "results": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                    "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                    "rate_limit_until": 0,
                    "total_posts": 0,
                    "total_replies": 0
                }
            
            async with asyncio.timeout(60):
                submissions = {s.id: s async for s in reddit.info(fullnames=ids)}
            
            for post_id in post_ids:
                if post_id not in submissions:
                    logger.error(f"無法獲取貼文：{post_id}")
                    rate_limit_info.append({"message": f"無法獲取貼文 {post_id}"})
                    continue
                submission = submissions[post_id]
                comments, comments_rate_limit_info = await collect_comments(submission, max_comments, sort)
                rate_limit_info.extend(comments_rate_limit_info)
                request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
                
                result = {
                    "title": submission.title,
                    "total_replies": submission.num_comments,
                    "last_reply_time": datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S"),
                    "like_count": submission.score,
                    "replies": comments,
                    "fetched_pages": [1],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": last_reset,
                    "rate_limit_until": 0,
                    "replies_count": len(comments)
                }
                results.append(result)
                total_posts += 1
                total_replies += len(comments)
                cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}_{sort}"
                client_manager.cache["thread"][cache_key] = {"timestamp": time.time(), "data": result}
            
            logger.info(f"批次抓取完成：帖子數={total_posts}, 回覆數={total_replies}")
            return {
                "results": results,
                "rate_limit_info": rate_limit_info,
                "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                "rate_limit_until": 0,
                "total_posts": total_posts,
                "total_replies": total_replies
            }
        except Exception as e:
            _handle_api_error(e, "批次抓取貼文", rate_limit_info)
            return {
                "results": results,
                "rate_limit_info": rate_limit_info,
                "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                "rate_limit_until": 0,
                "total_posts": total_posts,
                "total_replies": total_replies
            }

async def get_subreddit_name(subreddit):
    async with client_manager.get_client() as reddit:
        try:
            subreddit_obj = await reddit.subreddit(subreddit)
            return subreddit_obj.display_name
        except Exception as e:
            logger.error(f"獲取子版名稱失敗：{str(e)}", exc_info=True)
            return "未知子版"
