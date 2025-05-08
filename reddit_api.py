import asyncpraw
import logging
import asyncio
from datetime import datetime
import streamlit as st
import time
import pytz
from logging_config import configure_logger

HONG_KONG_TZ = pytz.timezone("Asia/Hong_Kong")

logger = configure_logger(__name__, "reddit_api.log")

request_counter = 0
last_reset = time.time()
RATE_LIMIT_REQUESTS_PER_MINUTE = 780
client_initialized = False

topic_cache = {}
thread_cache = {}
CACHE_DURATION = 300  # 緩存 5 分鐘
MAX_CACHE_SIZE = 100  # 最大緩存條目數

async def init_reddit_client():
    global request_counter, last_reset, client_initialized
    try:
        current_time = time.time()
        if current_time - last_reset >= 60:
            request_counter = 0
            last_reset = current_time
            logger.info("速率限制計數器重置")

        reddit = asyncpraw.Reddit(
            client_id=st.secrets["reddit"]["client_id"],
            client_secret=st.secrets["reddit"]["client_secret"],
            username=st.secrets["reddit"]["username"],
            password=st.secrets["reddit"]["password"],
            user_agent=f"LIHKGChatBot/v1.0 by u/{st.secrets['reddit']['username']}"
        )
        user = await reddit.user.me()
        if user and not client_initialized:
            logger.info(f"Reddit 客戶端初始化成功，已認證用戶：{user.name}")
            client_initialized = True
        elif user:
            logger.debug(f"Reddit 客戶端已初始化，重用認證用戶：{user.name}")
        else:
            logger.warning("Reddit 客戶端初始化成功，但未成功認證用戶，可能未獲得更高配額")
        return reddit
    except KeyError as e:
        logger.error(f"缺少 Secrets 配置：{str(e)}")
        raise
    except Exception as e:
        logger.error(f"Reddit 客戶端初始化失敗：{str(e)}")
        raise

async def get_subreddit_name(subreddit, reddit=None):
    if reddit is None:
        reddit = await init_reddit_client()
    try:
        subreddit_obj = await reddit.subreddit(subreddit)
        display_name = subreddit_obj.display_name
        return display_name
    except Exception as e:
        logger.error(f"獲取子版名稱失敗：{str(e)}")
        return "未知子版"
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()

def clean_cache(cache, cache_type="topic"):
    current_time = time.time()
    expired_keys = [key for key, value in cache.items() if current_time - value["timestamp"] > CACHE_DURATION]
    for key in expired_keys:
        del cache[key]
    if len(cache) > MAX_CACHE_SIZE:
        sorted_keys = sorted(cache.items(), key=lambda x: x[1]["timestamp"])
        for key, _ in sorted_keys[:len(cache) - MAX_CACHE_SIZE]:
            del cache[key]
        logger.info(f"清理 {cache_type} 緩存，移除 {len(sorted_keys[:len(cache) - MAX_CACHE_SIZE])} 個過舊條目，當前緩存大小：{len(cache)}")

async def get_reddit_topic_list(subreddit, start_page=1, max_pages=1, sort="best", reddit=None):
    global request_counter, last_reset, topic_cache
    cache_key = f"{subreddit}_{start_page}_{max_pages}_{sort}"
    
    clean_cache(topic_cache, "topic")
    if cache_key in topic_cache:
        cached_data = topic_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"使用緩存數據，子版：{subreddit}, 鍵：{cache_key}")
            return cached_data["data"]
    
    if reddit is None:
        reddit = await init_reddit_client()
    items = []
    rate_limit_info = []
    local_last_reset = last_reset
    
    try:
        total_limit = 100
        subreddit_obj = await reddit.subreddit(subreddit)
        request_counter += 1
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
            wait_time = 60 - (time.time() - local_last_reset)
            if wait_time > 0:
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                local_last_reset = time.time()
                last_reset = local_last_reset
                logger.info("速率限制計數器重置")
        
        logger.info(f"開始抓取子版 {subreddit}，排序：{sort}，當前請求次數 {request_counter}")
        
        if sort not in ["best", "new"]:
            raise ValueError(f"不支持的排序方式：{sort}，僅支援 'best' 或 'new'")
        
        if sort == "best":
            async for submission in subreddit_obj.top(time_filter="day", limit=total_limit):
                if submission.created_utc:
                    hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
                    readable_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
                    item = {
                        "thread_id": submission.id,
                        "title": submission.title,
                        "no_of_reply": submission.num_comments,
                        "last_reply_time": readable_time,
                        "like_count": submission.score
                    }
                    items.append(item)
        elif sort == "new":
            async for submission in subreddit_obj.new(limit=total_limit):
                if submission.created_utc:
                    hk_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ)
                    readable_time = hk_time.strftime("%Y-%m-%d %H:%M:%S")
                    item = {
                        "thread_id": submission.id,
                        "title": submission.title,
                        "no_of_reply": submission.num_comments,
                        "last_reply_time": readable_time,
                        "like_count": submission.score
                    }
                    items.append(item)
        
        logger.info(f"抓取子版 {subreddit} 成功，總項目數 {len(items)}")
        
        topic_cache[cache_key] = {
            "timestamp": time.time(),
            "data": {
                "items": items,
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": local_last_reset,
                "rate_limit_until": 0
            }
        }
    except Exception as e:
        logger.error(f"抓取貼文列表失敗：{str(e)}")
        rate_limit_info.append({"message": f"抓取子版 {subreddit} 失敗：{str(e)}"})
        return {
            "items": items,
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": 0
        }
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()
    
    return {
        "items": items,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": local_last_reset,
        "rate_limit_until": 0
    }

async def collect_more_comments(reddit, submission, max_comments, request_counter, local_last_reset):
    """收集 MoreComments 的 children ID，分批調用 /api/morechildren"""
    more_comments_ids = []
    comments = []
    
    async for comment in submission.comments:
        if isinstance(comment, asyncpraw.models.MoreComments):
            more_comments_ids.extend(comment.children)
        elif hasattr(comment, 'body') and comment.body:
            hk_time = datetime.fromtimestamp(comment.created_utc, tz=HONG_KONG_TZ)
            comments.append({
                "reply_id": comment.id,
                "msg": comment.body,
                "like_count": comment.score,
                "reply_time": hk_time.strftime("%Y-%m-%d %H:%M:%S")
            })
    
    BATCH_SIZE = 100
    for i in range(0, len(more_comments_ids), BATCH_SIZE):
        if len(comments) >= max_comments:
            break
        batch_ids = more_comments_ids[i:i + BATCH_SIZE]
        if not batch_ids:
            continue
        
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
            wait_time = 60 - (time.time() - local_last_reset)
            if wait_time > 0:
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                local_last_reset = time.time()
        
        try:
            more_comments = await reddit.comment.more_children(
                link_id=f"t3_{submission.id}",
                children=",".join(batch_ids),
                sort="best"
            )
            request_counter += 1
            for comment in more_comments:
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
            logger.warning(f"/api/morechildren 請求失敗：{str(e)}")
            continue
    
    return comments[:max_comments], request_counter, local_last_reset

async def get_reddit_thread_content(post_id, subreddit, max_comments=100, reddit=None):
    global request_counter, last_reset, thread_cache
    cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}"
    
    clean_cache(thread_cache, "thread")
    if cache_key in thread_cache:
        cached_data = thread_cache[cache_key]
        if time.time() - cached_data["timestamp"] < CACHE_DURATION:
            logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
            return cached_data["data"]
    
    if reddit is None:
        reddit = await init_reddit_client()
        logger.debug(f"重新初始化 Reddit 客戶端，post_id: {post_id}, reddit type: {type(reddit)}")
    else:
        logger.debug(f"使用傳入 Reddit 客戶端，post_id: {post_id}, reddit type: {type(reddit)}")
    
    replies = []
    rate_limit_info = []
    local_last_reset = last_reset
    
    try:
        request_counter += 1
        if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
            wait_time = 60 - (time.time() - local_last_reset)
            if wait_time > 0:
                logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                await asyncio.sleep(wait_time)
                request_counter = 0
                local_last_reset = time.time()
                last_reset = local_last_reset
                logger.info("速率限制計數器重置")
        
        logger.info(f"開始抓取貼文：[{post_id}]，當前請求次數：{request_counter}")
        
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
                "last_reset": local_last_reset,
                "rate_limit_until": 0
            }
        
        title = submission.title
        total_replies = submission.num_comments
        like_count = submission.score
        last_reply_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S")
        
        replies, request_counter, local_last_reset = await collect_more_comments(
            reddit, submission, max_comments, request_counter, local_last_reset
        )
        
        logger.info(f"抓取貼文完成：{{{post_id}: {len(replies)}}}，總回覆數：{len(replies)}")
        
        thread_cache[cache_key] = {
            "timestamp": time.time(),
            "data": {
                "title": title,
                "total_replies": total_replies,
                "last_reply_time": last_reply_time,
                "like_count": like_count,
                "replies": replies,
                "fetched_pages": [1],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": local_last_reset,
                "rate_limit_until": 0
            }
        }
    except Exception as e:
        logger.error(f"抓取貼文內容失敗：{str(e)}")
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
            "last_reset": local_last_reset,
            "rate_limit_until": 0
        }
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()
    
    return {
        "title": title,
        "total_replies": total_replies,
        "last_reply_time": last_reply_time,
        "like_count": like_count,
        "replies": replies,
        "fetched_pages": [1],
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": local_last_reset,
        "rate_limit_until": 0
    }

async def get_reddit_thread_content_batch(post_ids, subreddit, max_comments=100):
    global request_counter, last_reset, thread_cache
    results = []
    rate_limit_info = []
    local_last_reset = last_reset
    fetch_status = {}
    
    reddit = await init_reddit_client()
    try:
        logger.info(f"開始抓取貼文：{post_ids}，當前請求次數：{request_counter}")
        
        for post_id in post_ids:
            cache_key = f"{post_id}_subreddit_{subreddit}_{max_comments}"
            clean_cache(thread_cache, "thread")
            if cache_key in thread_cache:
                cached_data = thread_cache[cache_key]
                if time.time() - cached_data["timestamp"] < CACHE_DURATION:
                    logger.info(f"使用緩存數據，貼文：{post_id}, 鍵：{cache_key}")
                    results.append(cached_data["data"])
                    fetch_status[post_id] = {"status": "cached", "replies": len(cached_data["data"]["replies"])}
                    continue
            
            request_counter += 1
            if request_counter >= RATE_LIMIT_REQUESTS_PER_MINUTE:
                wait_time = 60 - (time.time() - local_last_reset)
                if wait_time > 0:
                    logger.warning(f"達到速率限制，等待 {wait_time:.2f} 秒")
                    await asyncio.sleep(wait_time)
                    request_counter = 0
                    local_last_reset = time.time()
                    last_reset = local_last_reset
                    logger.info("速率限制計數器重置")
            
            submission = await reddit.submission(id=post_id)
            if not submission:
                logger.error(f"無法獲取貼文：{post_id}")
                rate_limit_info.append({"message": f"無法獲取貼文 {post_id}"})
                results.append({
                    "title": "",
                    "total_replies": 0,
                    "last_reply_time": "1970-01-01 00:00:00",
                    "like_count": 0,
                    "replies": [],
                    "fetched_pages": [],
                    "rate_limit_info": rate_limit_info,
                    "request_counter": request_counter,
                    "last_reset": local_last_reset,
                    "rate_limit_until": 0
                })
                fetch_status[post_id] = {"status": "failed", "replies": 0}
                continue
            
            title = submission.title
            total_replies = submission.num_comments
            like_count = submission.score
            last_reply_time = datetime.fromtimestamp(submission.created_utc, tz=HONG_KONG_TZ).strftime("%Y-%m-%d %H:%M:%S")
            
            replies, request_counter, local_last_reset = await collect_more_comments(
                reddit, submission, max_comments, request_counter, local_last_reset
            )
            
            result = {
                "title": title,
                "total_replies": total_replies,
                "last_reply_time": last_reply_time,
                "like_count": like_count,
                "replies": replies,
                "fetched_pages": [1],
                "rate_limit_info": rate_limit_info,
                "request_counter": request_counter,
                "last_reset": local_last_reset,
                "rate_limit_until": 0
            }
            results.append(result)
            fetch_status[post_id] = {"status": "success", "replies": len(replies)}
            
            thread_cache[cache_key] = {
                "timestamp": time.time(),
                "data": result
            }
        
        total_replies = sum(status["replies"] for status in fetch_status.values() if status["status"] in ["success", "cached"])
        logger.info(f"抓取貼文完成：{fetch_status}，總回覆數：{total_replies}")
        
    except Exception as e:
        logger.error(f"批次抓取貼文內容失敗：{str(e)}")
        rate_limit_info.append({"message": f"批次抓取貼文失敗：{str(e)}"})
        results.append({
            "title": "",
            "total_replies": 0,
            "last_reply_time": "1970-01-01 00:00:00",
            "like_count": 0,
            "replies": [],
            "fetched_pages": [],
            "rate_limit_info": rate_limit_info,
            "request_counter": request_counter,
            "last_reset": local_last_reset,
            "rate_limit_until": 0
        })
        fetch_status[post_id] = {"status": "failed", "replies": 0}
    finally:
        if reddit and not hasattr(reddit, 'is_shared'):
            await reddit.close()
    
    return {
        "results": results,
        "rate_limit_info": rate_limit_info,
        "request_counter": request_counter,
        "last_reset": local_last_reset,
        "rate_limit_until": 0
    }
