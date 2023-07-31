import requests
from pyrate_limiter import MemoryListBucket
from requests_cache import CacheMixin, FileCache
from requests_ratelimiter import LimiterMixin


class CachedLimiterSession(CacheMixin, LimiterMixin, requests.Session):
    pass

    @staticmethod
    def new(
        name: str,
        per_second: int = 1,
        expire_after="NEVER_EXPIRE",
    ) -> "CachedLimiterSession":
        return CachedLimiterSession(
            backend=FileCache(name),
            bucket_class=MemoryListBucket,
            expire_after=expire_after,
            per_second=per_second,
        )
