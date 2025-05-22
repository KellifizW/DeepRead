import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from contextlib import asynccontextmanager
from logging_config import configure_logger
import aiohttp

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")
logger = configure_logger(__name__, "reddit_api.log")

class RedditClientManager:
    def __init__(self):
        try:
            credentials = st.secrets["reddit"]["credentials"]
            if not credentials or len(credentials) < 1:
                raise KeyError("Reddit credentials are missing or empty")
            self.clients = [None] * len(credentials)
            self.request_counters = [0] * len(self.clients)
            self.last_resets = [time.time()] * len(self.clients)
            self.rate_limit_requests_per_minute = 60  # 提高到 Reddit API 標準限制（600 次/10 分鐘 ≈ 60 次/分鐘）
            self.current_client_index = 0
            self.topic_cache = {}
            self.thread_cache = {}
            self.topic_cache_duration = 1800
            self.thread_cache_duration = 600
            self.max_cache_size = 100
        except KeyError as e:
            logger.error(f"Secrets 配置錯誤：{str(e)}")
            raise

    @asynccontextmanager
    async def get_client(self):
        for i in range(len(self.clients)):
            client_idx = (self.current_client_index + i) % len(self.clients)
            if self.clients[client_idx] is None:
                try:
                    self.clients[client_idx] = await self._init_reddit_client(client_idx)
                except Exception as e:
                    logger.error(f"客戶端 {client_idx} 初始化失敗：{str(e)}，嘗試下一個客戶端", exc_info=True)
                    continue
            try:
                self.request_counters[client_idx], self.last_resets[client_idx] = await self._handle_rate_limit(client_idx)
                self.current_client_index = client_idx
                yield self.clients[client_idx]
                return
            except Exception as e:
                logger.error(f"客戶端 {client_idx} 獲取失敗：{str(e)}，嘗試下一個客戶端", exc_info=True)
                continue
        raise Exception("所有 Reddit 客戶端均無法初始化")

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
            reddit.is_shared = True
            try:
                user = await reddit.user.me()
                logger.info(f"Reddit 客戶端 {client_idx} 初始化成功，已認證用戶：{user.name}")
                try:
                    response = await reddit.get("api/v1/me")
                    logger.debug(f"客戶端 {client_idx} API 回應：{response}")
                    # 嘗試從回應中提取速率限制信息
                    headers = response.get("headers", {}) if isinstance(response, dict) else getattr(response, 'headers', {})
                    remaining = headers.get("X-Ratelimit-Remaining", "未知")
                    reset_time = headers.get("X-Ratelimit-Reset", "未知")
                    if remaining == "未知" and isinstance(response, dict):
                        logger.info(f"客戶端 {client_idx} 速率限制資訊：無法直接獲取 headers，回應為字典")
                    else:
                        logger.info(
                            f"客戶端 {client_idx} 速率限制資訊："
                            f"X-Ratelimit-Remaining={remaining}, "
                            f"X-Ratelimit-Reset={reset_time}"
                        )
                except Exception as e:
                    logger.warning(f"客戶端 {client_idx} 無法獲取速率限制資訊：{str(e)}")
                return reddit
            except asyncpraw.exceptions.RedditAPIException as e:
                logger.error(
                    f"Reddit API 錯誤（客戶端 {client_idx}）：{str(e)}, "
                    f"詳細：{e.__dict__}, "
                    f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                    f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
                )
                raise
            except Exception as e:
                logger.error(f"Reddit 客戶端 {client_idx} 初始化失敗：{str(e)}", exc_info=True)
                raise
        except KeyError as e:
            logger.error(f"缺少 Secrets 配置（客戶端 {client_idx}）：{str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Reddit 客戶端 {client_idx} 初始化失敗：{str(e)}, "
                f"可能原因：憑證無效、帳戶受限或網路問題", exc_info=True
            )
            raise

    async def _handle_rate_limit(self, client_idx):
        current_time = time.time()
        if self.request_counters[client_idx] >= self.rate_limit_requests_per_minute:
            wait_time = 60 - (current_time - self.last_resets[client_idx])
            if wait_time > 0:
                logger.warning(f"客戶端 {client_idx} 達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                self.request_counters[client_idx] = 0
                self.last_resets[client_idx] = current_time
                logger.info(f"客戶端 {client_idx} 速率限制計數器重置")
        self.request_counters[client_idx] += 1
        return self.request_counters[client_idx], self.last_resets[client_idx]

    def clean_cache(self, cache, cache_type="topic"):
        duration = self.topic_cache_duration if cache_type == "topic" else self.thread_cache_duration
        current_time = time.time()
        expired_keys = [k for k, v in cache.items() if current_time - v["timestamp"] > duration]
        for k in expired_keys:
            del cache[k]
        if len(cache) > self.max_cache_size:
            sorted_keys = sorted(cache.items(), key=lambda x: x[1]["timestamp"])[:len(cache) - self.max_cache_size]
            for k, _ in sorted_keys:
                del cache[k]
            logger.info(f"清理 {cache_type} 緩存，移除 {len(sorted_keys)} 個條目，當前緩存大小：{len(cache)}")

client_manager = RedditClientManager()

async def get_subreddit_name(subreddit):
    async with client_manager.get_client() as reddit:
        try:
            subreddit_obj = await reddit.subreddit(subreddit)
            return subreddit_obj.display_name
        except asyncpraw.exceptions.RedditAPIException as e:
            logger.error(
                f"獲取子版名稱失敗：{str(e)}, "
                f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
            )
            return "未知子版"
        except Exception as e:
            logger.error(f"獲取子版名稱失敗：{str(e)}", exc_info=True)
            return "未知子版"

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
    client_manager.clean_cache(client_manager.topic_cache, "topic")
    
    if cache_key in client_manager.topic_cache:
        cached_data = client_manager.topic_cache[cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.topic_cache_duration:
            logger.info(f"使用緩存數據，子版：{subreddit}, 鍵：{cache_key}")
            return cached_data["data"]
    
    items = []
    rate_limit_info = []
    total_limit = 250
    items_per_page = 100
    start_index = (start_page - 1) * items_per_page
    end_index = start_index + (max_pages * items_per_page)
    
    # 嘗試每個客戶端直到成功
    for client_idx in range(len(client_manager.clients)):
        async with client_manager.get_client() as reddit:
            for attempt in range(3):
                try:
                    subreddit_obj = await reddit.subreddit(subreddit)
                    sort_methods = {
                        "confidence": lambda: subreddit_obj.hot(limit=total_limit),
                        "new": lambda: subreddit_obj.new(limit=total_limit),
                        "top": lambda: subreddit_obj.top(time_filter="day", limit=total_limit),
                        "controversial": lambda: subreddit_obj.controversial(time_filter="day", limit=total_limit),
                        "rising": lambda: subreddit_obj.rising(limit=total_limit)
                    }
                    
                    if sort not in sort_methods:
                        logger.warning(f"不支持的排序方式：{sort}，使用默認排序：confidence")
                        sort = "confidence"
                    
                    logger.info(f"開始抓取子版 {subreddit}，排序：{sort}，嘗試：{attempt + 1}，客戶端：{client_idx}，當前請求次數 {client_manager.request_counters[client_idx]}")
                    
                    async def fetch_submissions():
                        submissions = []
                        async for submission in sort_methods[sort]():
                            submissions.append(submission)
                        return submissions
                    
                    async with asyncio.timeout(60):
                        submissions = await fetch_submissions()
                    
                    for idx, submission in enumerate(submissions):
                        if idx < start_index or idx >= end_index:
                            continue
                        try:
                            async with asyncio.timeout(10):
                                if submission.created_utc:
                                    formatted_submission = await format_submission(submission)
                                    items.append(formatted_submission)
                        except asyncio.TimeoutError:
                            logger.error(f"抓取貼文超時：子版={subreddit}, 貼文ID={getattr(submission, 'id', 'unknown')}")
                            rate_limit_info.append({"message": f"抓取貼文超時：子版={subreddit}"})
                            continue
                        except asyncpraw.exceptions.RedditAPIException as e:
                            logger.error(
                                f"處理貼文失敗：子版={subreddit}, 貼文ID={getattr(submission, 'id', 'unknown')}, "
                                f"錯誤={str(e)}, "
                                f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                                f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
                            )
                            rate_limit_info.append({"message": f"處理貼文失敗：子版={subreddit}, 錯誤={str(e)}"})
                            continue
                        except Exception as e:
                            logger.error(f"處理貼文失敗：子版={subreddit}, 貼文ID={getattr(submission, 'id', 'unknown')}, 錯誤={str(e)}", exc_info=True)
                            rate_limit_info.append({"message": f"處理貼文失敗：子版={subreddit}, 錯誤={str(e)}"})
                            continue
                    
                    logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
                    
                    client_manager.topic_cache[cache_key] = {
                        "timestamp": time.time(),
                        "data": {
                            "items": items,
                            "rate_limit_info": rate_limit_info,
                            "request_counter": client_manager.request_counters[client_idx],
                            "last_reset": client_manager.last_resets[client_idx],
                            "rate_limit_until": 0,
                            "total_posts": len(items)
                        }
                    }
                    return {
                        "items": items,
                        "rate_limit_info": rate_limit_info,
                        "request_counter": client_manager.request_counters[client_idx],
                        "last_reset": client_manager.last_resets[client_idx],
                        "rate_limit_until": 0,
                        "total_posts": len(items)
                    }
                
                except asyncprawcore.exceptions.RequestException as e:
                    logger.error(f"抓取貼文列表失敗（第 {attempt + 1} 次嘗試，客戶端 {client_idx}）：子版={subreddit}, 錯誤={str(e)}", exc_info=True)
                    if "Session is closed" in str(e) and attempt < 2:
                        logger.info(f"檢測到 Session is closed 錯誤，等待 2 秒後重試")
                        await asyncio.sleep(2)
                        continue
                    rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
                    break
                except asyncio.TimeoutError:
                    logger.error(f"抓取貼文列表超時（第 {attempt + 1} 次嘗試，客戶端 {client_idx}）：子版={subreddit}", exc_info=True)
                    rate_limit_info.append({"message": f"抓取子版 {subreddit} 超時"})
                    break
                except asyncpraw.exceptions.RedditAPIException as e:
                    logger.error(
                        f"抓取貼文列表失敗（第 {attempt + 1} 次嘗試，客戶端 {client_idx}）：子版={subreddit}, "
                        f"錯誤={str(e)}, "
                        f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                        f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
                    )
                    rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
                    break
                except Exception as e:
                    logger.error(f"抓取貼文列表失敗（第 {attempt + 1} 次嘗試，客戶端 {client_idx}）：子版={subreddit}, 錯誤={str(e)}", exc_info=True)
                    rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
                    break
    
    logger.warning(f"所有客戶端均無法抓取子版 {subreddit}，返回空結果")
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
        "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
        "rate_limit_until": 0,
        "total_posts": len(items)
    }

async def collect_more_comments(reddit, submission, max_comments, request_counter, last_reset, sort="confidence"):
    comments = []
    max_more_comments = 100
    
    try:
        # 在加載評論之前設置排序
        submission.comment_sort = sort
        logger.info(f"設置貼文 {submission.id} 的評論排序為 {sort}")
        
        # 清空現有評論並重新加載
        submission._comments = None
        await submission.load()
        
        for attempt in range(3):
            try:
                async with asyncio.timeout(60):
                    logger.info(f"開始展開 MoreComments，貼文ID={submission.id}，嘗試={attempt + 1}")
                    start_time = time.time()
                    await submission.comments.replace_more(limit=max_more_comments)
                    logger.info(f"展開 MoreComments 完成，貼文ID={submission.id}，耗時={time.time() - start_time:.2f} 秒")
                    break
            except asyncio.TimeoutError:
                logger.warning(f"展開 MoreComments 超時，貼文ID={submission.id}，嘗試={attempt + 1}")
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                logger.error(f"展開 MoreComments 最終超時：貼文ID={submission.id}")
                return comments[:max_comments], request_counter, last_reset
            except asyncpraw.exceptions.RedditAPIException as e:
                logger.error(
                    f"展開 MoreComments 失敗：貼文ID={submission.id}, "
                    f"錯誤={str(e)}, "
                    f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                    f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
                )
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return comments[:max_comments], request_counter, last_reset
            except Exception as e:
                logger.error(f"展開 MoreComments 失敗：貼文ID={submission.id}，錯誤={str(e)}，嘗試={attempt + 1}", exc_info=True)
                if attempt < 2:
                    await asyncio.sleep(2)
                    continue
                return comments[:max_comments], request_counter, last_reset
        
        # 檢查是否接近速率限制
        if request_counter >= client_manager.rate_limit_requests_per_minute - 5:
            wait_time = 60 - (time.time() - last_reset)
            if wait_time > 0:
                logger.warning(f"接近速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                last_reset = time.time()
        
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
            except asyncio.TimeoutError:
                logger.error(f"抓取評論超時：貼文ID={submission.id}, 評論ID={getattr(comment, 'id', 'unknown')}")
                continue
            except asyncpraw.exceptions.RedditAPIException as e:
                logger.error(
                    f"處理評論失敗：貼文ID={submission.id}, 評論ID={getattr(comment, 'id', 'unknown')}, "
                    f"錯誤={str(e)}, "
                    f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                    f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
                )
                continue
            except Exception as e:
                logger.error(f"處理評論失敗：貼文ID={submission.id}, 評論ID={getattr(comment, 'id', 'unknown')}, 錯誤={str(e)}", exc_info=True)
                continue
        
        request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
        
        return comments[:max_comments], request_counter, last_reset
    
    except asyncpraw.exceptions.RedditAPIException as e:
        logger.error(
            f"收集評論失敗：貼文ID={submission.id}, "
            f"錯誤={str(e)}, "
            f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
            f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
        )
        return comments[:max_comments], request_counter, last_reset
    except Exception as e:
        logger.error(f"收集評論失敗：貼文ID={submission.id}, 錯誤={str(e)}", exc_info=True)
        return comments[:max_comments], request_counter, last_reset

async def fetch_single_thread(post_id, subreddit, max_comments, reddit, request_counter, last_reset, sort="confidence"):
    if not post_id or not post_id.strip() or post_id.lower() == "id":
        logger.error(f"無效的貼文 ID：{post_id}")
        rate_limit_info = [{"message": f"無效的貼文 ID：{post_id}"}]
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
        }, request_counter, last_reset

    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}_{sort}"
    client_manager.clean_cache(client_manager.thread_cache, "thread")
    
    if cache_key in client_manager.thread_cache:
        cached_data = client_manager.thread_cache[cache_key]
        if time.time() - cached_data["timestamp"] < client_manager.thread_cache_duration:
            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
            return cached_data["data"], request_counter, last_reset
    
    replies = []
    rate_limit_info = []
    
    try:
        request_counter, last_reset = await client_manager._handle_rate_limit(client_manager.current_client_index % len(client_manager.clients))
        
        logger.info(f"開始抓取貼文：[{post_id}]，排序：{sort}，當前請求次數：{request_counter}")
        
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
            }, request_counter, last_reset
        
        title = submission.title
        total_replies = submission.num_comments
        like_count = submission.score
        last_reply_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S")
        
        replies, request_counter, last_reset = await collect_more_comments(
            reddit, submission, max_comments, request_counter, last_reset, sort=sort
        )
        
        logger.info(f"抓取貼文完成：{{{post_id}: {len(replies)}}}，總回覆數：{len(replies)}")
        
        result = {
            "title": title,
            "total_replies": total_replies,
            "last_reply_time": last_reply_time,
            "like_count": like_count,
            "replies": replies,
            "fetched_pages": [1],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": last_reset,
            "rate_limit_until": 0,
            "replies_count": len(replies)
        }
        
        client_manager.thread_cache[cache_key] = {
            "timestamp": time.time(),
            "data": result
        }
        
        return result, request_counter, last_reset
    
    except asyncio.TimeoutError as e:
        logger.error(f"抓取貼文內容超時：貼文ID={post_id}, 錯誤={str(e)}")
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 超時：{str(e)}"})
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
        }, request_counter, last_reset
    except aiohttp.ClientResponseError as e:
        logger.error(
            f"抓取貼文內容失敗：貼文ID={post_id}, "
            f"錯誤={str(e)}, "
            f"狀態碼={e.status}, "
            f"回應頭部：{e.headers or '無頭部資訊'}, "
            f"回應正文：{e.message or '無正文資訊'}"
        )
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
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
        }, request_counter, last_reset
    except asyncpraw.exceptions.RedditAPIException as e:
        logger.error(
            f"抓取貼文內容失敗：貼文ID={post_id}, "
            f"錯誤={str(e)}, "
            f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
            f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
        )
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
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
        }, request_counter, last_reset
    except Exception as e:
        logger.error(f"抓取貼文內容失敗：貼文ID={post_id}, 錯誤={str(e)}", exc_info=True)
        rate_limit_info.append({"message": f"抓取貼文 {post_id} 失敗：{str(e)}"})
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
        }, request_counter, last_reset

async def get_reddit_thread_content(post_id, subreddit, max_comments=100, sort="confidence"):
    async with client_manager.get_client() as reddit:
        try:
            result, request_counter, last_reset = await fetch_single_thread(
                post_id, subreddit, max_comments, reddit, 
                client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                sort=sort
            )
            return result
        except Exception as e:
            logger.error(f"抓取單篇貼文內容失敗：貼文ID={post_id}, 錯誤={str(e)}", exc_info=True)
            raise

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=100, sort="confidence"):
    results = []
    rate_limit_info = []
    fetch_status = {}
    batch_size = 5
    total_posts = 0
    total_replies = 0
    
    async with client_manager.get_client() as reddit:
        try:
            logger.info(f"開始批次抓取貼文：{post_ids}，排序：{sort}")
            
            ids = [f"t3_{pid}" for pid in post_ids if pid and pid.strip() and pid.lower() != "id"]
            if not ids:
                logger.error(f"無有效的貼文 ID：{post_ids}")
                rate_limit_info.append({"message": "無有效的貼文 ID"})
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
            
            for i in range(0, len(post_ids), batch_size):
                batch_ids = [pid for pid in post_ids[i:i + batch_size] if pid and pid.strip() and pid.lower() != "id"]
                tasks = []
                for post_id in batch_ids:
                    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}_{sort}"
                    client_manager.clean_cache(client_manager.thread_cache, "thread")
                    if cache_key in client_manager.thread_cache:
                        cached_data = client_manager.thread_cache[cache_key]
                        if time.time() - cached_data["timestamp"] < client_manager.thread_cache_duration:
                            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
                            results.append(cached_data["data"])
                            fetch_status[post_id] = {"status": "cached", "replies": len(cached_data["data"]["replies"])}
                            total_posts += 1
                            total_replies += len(cached_data["data"]["replies"])
                            continue
                    tasks.append(fetch_single_thread(
                        post_id, subreddit, max_comments, reddit,
                        client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
                        client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
                        sort=sort
                    ))
                
                if tasks:
                    batch_results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in batch_results:
                        if isinstance(result, tuple):
                            result, req_count, last_reset = result
                            results.append(result)
                            fetch_status[result.get("title", post_id)] = {
                                "status": "success",
                                "replies": len(result.get("replies", []))
                            }
                            rate_limit_info.extend(result.get("rate_limit_info", []))
                            client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)] = req_count
                            client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)] = last_reset
                            total_posts += 1
                            total_replies += len(result.get("replies", []))
                        else:
                            logger.error(
                                f"批次抓取貼文失敗：{str(result)}, "
                                f"回應頭部：{getattr(result, 'response', {}).get('headers', '無頭部資訊')}, "
                                f"回應正文：{getattr(result, 'response', {}).get('text', '無正文資訊')}"
                            )
                            rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(result)}"})
                            results.append({
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
                            })
                            fetch_status[post_id] = {"status": "failed", "replies": 0}
            
            total_requests = sum(client_manager.request_counters)
            logger.info(f"Reddit API 調用統計：抓取帖子數={total_posts}, 回覆數={total_replies}, 總請求次數={total_requests}")
        
        except asyncio.TimeoutError:
            logger.error(f"批次抓取貼文超時：{post_ids}", exc_info=True)
            rate_limit_info.append({"message": f"批次抓取貼文超時：{post_ids}"})
        except asyncpraw.exceptions.RedditAPIException as e:
            logger.error(
                f"批次抓取貼文內容失敗：{str(e)}, "
                f"回應頭部：{getattr(e, 'response', {}).get('headers', '無頭部資訊')}, "
                f"回應正文：{getattr(e, 'response', {}).get('text', '無正文資訊')}"
            )
            rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(e)}"})
        except Exception as e:
            logger.error(f"批次抓取貼文內容失敗：{str(e)}", exc_info=True)
            rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(e)}"})
        
        return {
            "results": results,
            "rate_limit_info": rate_limit_info,
            "request_counter": client_manager.request_counters[client_manager.current_client_index % len(client_manager.clients)],
            "last_reset": client_manager.last_resets[client_manager.current_client_index % len(client_manager.clients)],
            "rate_limit_until": 0,
            "total_posts": total_posts,
            "total_replies": total_replies
        }
