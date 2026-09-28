import asyncio
import logging
import os

import aiohttp

from miles.utils.misc import SingletonMeta

logger = logging.getLogger(__name__)

_TIMEOUT = aiohttp.ClientTimeout(connect=10, sock_read=30, total=45)

# 设置代理
os.environ["http_proxy"], os.environ["https_proxy"] = "", ""


class RewardSessionManager(metaclass=SingletonMeta):
    """
    单例模式管理 Reward 的全局网络 Session 和限流 Semaphore。
    """

    def __init__(self):
        # 按事件循环缓存 session 和锁，避免跨 loop 共享导致 File descriptor 错误
        self._sessions: dict = {}
        self.inflight_sem = asyncio.Semaphore(int(os.getenv("OPD_RM_MAX_INFLIGHT", "64")))

    async def get_session(self) -> aiohttp.ClientSession:
        loop = asyncio.get_running_loop()

        if loop not in self._sessions:
            self._sessions[loop] = {"lock": asyncio.Lock(), "session": None}

        session_data = self._sessions[loop]

        if session_data["session"] is not None and not session_data["session"].closed:
            return session_data["session"]

        async with session_data["lock"]:
            if session_data["session"] is None or session_data["session"].closed:
                connector = aiohttp.TCPConnector(
                    limit=int(os.getenv("OPD_RM_CONN_LIMIT", "64")),
                    limit_per_host=int(os.getenv("OPD_RM_CONN_PER_HOST", "16")),
                    ttl_dns_cache=300,
                    enable_cleanup_closed=False,  # True triggers aiohttp fd-reuse bug → SIGABRT
                    force_close=True,
                )
                session_data["session"] = aiohttp.ClientSession(
                    timeout=_TIMEOUT,
                    connector=connector,
                    trust_env=False,  # 内网服务，避免走环境代理
                )
        return session_data["session"]

    async def reset_session(self) -> None:
        """关闭当前事件循环对应的 session，下次 get_session 时会自动重建。"""
        loop = asyncio.get_running_loop()
        session_data = self._sessions.get(loop)
        if session_data is None:
            return

        async with session_data["lock"]:
            old_session = session_data["session"]
            session_data["session"] = None
            if old_session is not None and not old_session.closed:
                await old_session.close()
                logger.info("Reward session reset: old session closed for current event loop.")
