import aiohttp
import json
import re
import random
from typing import AsyncGenerator, Dict, List, Optional

class Grok3Client:
    def __init__(self):
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.max_tokens = 100000

    async def analyze_question(self, user_query: str, cat_name: str, cat_id: int, **kwargs) -> Dict:
        """分析問題性質"""
        prompt = self._build_analysis_prompt(user_query, cat_name, cat_id, **kwargs)
        response = await self._call_api(prompt)
        return self._parse_analysis_response(response, cat_id)

    async def screen_threads(self, user_query: str, threads: List[Dict]) -> Dict:
        """篩選相關主題"""
        prompt = self._build_screening_prompt(user_query, threads)
        response = await self._call_api(prompt)
        return self._parse_screening_response(response, threads)

    async def stream_response(
        self,
        user_query: str,
        metadata: List[Dict],
        thread_data: Dict,
        processing: str
    ) -> AsyncGenerator[str, None]:
        """串流生成回應"""
        prompt = self._build_response_prompt(user_query, metadata, thread_data, processing)
        async for chunk in self._stream_api(prompt):
            yield chunk

    def _build_analysis_prompt(self, user_query: str, cat_name: str, cat_id: int, **kwargs) -> str:
        """構建分析提示"""
        # 簡化後的提示模板
        return f"""分析問題並返回JSON:
        - 問題: {user_query}
        - 分類: {cat_name}({cat_id})
        - 需要: theme, data_type, post_limit, reply_limit, processing
        - 示例: {{"theme":"搞笑","data_type":"both","post_limit":2,"reply_limit":75,"processing":"humor_focused_summary"}}
        """

    def _build_screening_prompt(self, user_query: str, threads: List[Dict]) -> str:
        """構建篩選提示"""
        return f"""篩選最相關的3個主題:
        - 問題: {user_query}
        - 候選主題: {json.dumps(threads[:50], ensure_ascii=False)}
        - 返回格式: {{"top_thread_ids":[],"need_replies":true,"reason":"..."}}
        """

    def _build_response_prompt(self, user_query: str, metadata: List[Dict], thread_data: Dict, processing: str) -> str:
        """構建回應提示"""
        theme_map = {
            "emotion_focused_summary": "感動",
            "humor_focused_summary": "搞笑", 
            "professional_summary": "專業"
        }
        return f"""根據以下數據生成300-500字回應:
        - 問題: {user_query}
        - 主題: {theme_map.get(processing, "一般")}
        - 元數據: {json.dumps(metadata, ensure_ascii=False)}
        - 內容: {json.dumps(thread_data, ensure_ascii=False)[:self.max_tokens//2]}
        - 要求: 使用繁體中文，重點突出{theme_map.get(processing, "")}內容
        """

    async def _call_api(self, prompt: str) -> Dict:
        """調用Grok3 API"""
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._get_api_key()}"
            }
            payload = {
                "model": "grok-3-beta",
                "messages": [
                    {"role": "system", "content": "你是一個智能助手"},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 600,
                "temperature": 0.7
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as resp:
                    data = await resp.json()
                    return data["choices"][0]["message"]["content"]
                    
        except Exception as e:
            print(f"Grok3 API error: {str(e)}")
            return {}

    async def _stream_api(self, prompt: str) -> AsyncGenerator[str, None]:
        """串流調用API"""
        try:
            headers = {
                "Content-Type": "application/json", 
                "Authorization": f"Bearer {self._get_api_key()}"
            }
            payload = {
                "model": "grok-3-beta",
                "messages": [
                    {"role": "system", "content": "你是一個智能助手"},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "max_tokens": 1000,
                "temperature": 0.7
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(self.api_url, headers=headers, json=payload) as resp:
                    async for line in resp.content:
                        if line.startswith(b"data: "):
                            try:
                                chunk = json.loads(line[6:])
                                content = chunk["choices"][0]["delta"].get("content", "")
                                if content:
                                    yield content
                            except:
                                continue

        except Exception as e:
            yield f"錯誤: {str(e)}"

    def _get_api_key(self) -> str:
        """從環境變量獲取API密鑰"""
        import os
        return os.getenv("GROK3_API_KEY", "default_key")

    def _parse_analysis_response(self, response: str, cat_id: int) -> Dict:
        """解析分析回應"""
        try:
            result = json.loads(response.strip())
            result["category_ids"] = [cat_id]  # 確保包含正確分類
            return result
        except:
            return {
                "theme": "一般",
                "category_ids": [cat_id],
                "data_type": "both",
                "post_limit": 2,
                "reply_limit": 75,
                "processing": "summarize"
            }

    def _parse_screening_response(self, response: str, threads: List[Dict]) -> Dict:
        """解析篩選回應"""
        try:
            result = json.loads(response.strip())
            valid_ids = {str(t["thread_id"]) for t in threads}
            
            # 驗證返回的ID是否有效
            result["top_thread_ids"] = [
                tid for tid in result.get("top_thread_ids", [])
                if str(tid) in valid_ids
            ]
            
            if not result["top_thread_ids"]:
                result["top_thread_ids"] = [t["thread_id"] for t in random.sample(threads, min(3, len(threads)))]
                result["reason"] = "自動選擇隨機主題"
                
            return result
        except:
            return {
                "top_thread_ids": [t["thread_id"] for t in threads[:3]],
                "need_replies": True,
                "reason": "解析失敗，使用前3個主題"
            }