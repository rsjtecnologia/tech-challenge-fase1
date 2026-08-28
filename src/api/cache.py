"""
Sistema de cache em memória para predições.

Usa cachetools.TTLCache para armazenar resultados de predições
com expiração automática baseada em TTL (time-to-live).

Thread-safe, sem dependências externas.
Inclui decorator @cached para reutilização em serviços.
"""
import functools
import hashlib
import json
import logging
from typing import Any, Callable, Optional

from cachetools import TTLCache

logger = logging.getLogger(__name__)


class PredictionCache:
    """Cache TTL para predições.

    Características:
    - Cache LRU + TTL: remove entradas mais antigas ou menos usadas
    - Thread-safe (TTLCache usa Lock internamente)
    - Métricas de hit/miss para monitoramento
    - Suporta chaves por features+modelo ou hash de bytes
    """

    def __init__(self, maxsize: int = 512, ttl: int = 300):
        """
        Args:
            maxsize: Número máximo de entradas no cache (LRU)
            ttl: Tempo de vida em segundos (padrão: 5 minutos)
        """
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0
        self._total = 0
        self._maxsize = maxsize
        self._ttl = ttl
        logger.info("Cache inicializado: maxsize=%d, ttl=%ds", maxsize, ttl)

    @staticmethod
    def make_key(features: list, model_name: str) -> str:
        """Gera chave hash para (features, model_name)."""
        rounded = [round(f, 6) if isinstance(f, float) else f for f in features]
        raw = json.dumps({"f": rounded, "m": model_name}, sort_keys=True)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def make_bytes_key(data: bytes, prefix: str = "img") -> str:
        """Gera chave hash para dados binários (imagens)."""
        return f"{prefix}:" + hashlib.md5(data).hexdigest()

    def get(self, features: list, model_name: str) -> Optional[Any]:
        """Retorna resultado cacheado por features+modelo."""
        return self.get_by_key(self.make_key(features, model_name))

    def set(self, features: list, model_name: str, value: Any) -> None:
        """Armazena resultado por features+modelo."""
        self.set_by_key(self.make_key(features, model_name), value)

    def get_by_key(self, key: str) -> Optional[Any]:
        """Retorna resultado cacheado por chave genérica."""
        result = self._cache.get(key)
        self._total += 1
        if result is not None:
            self._hits += 1
            logger.debug("CACHE HIT: %s...", key[:16])
        else:
            self._misses += 1
            logger.debug("CACHE MISS: %s...", key[:16])
        return result

    def set_by_key(self, key: str, value: Any) -> None:
        """Armazena resultado por chave genérica."""
        self._cache[key] = value
        logger.debug("CACHE SET: %s...", key[:16])

    def invalidate(self, features: list, model_name: str) -> None:
        """Remove entrada específica do cache por features+modelo."""
        key = self.make_key(features, model_name)
        self._cache.pop(key, None)

    def invalidate_by_key(self, key: str) -> None:
        """Remove entrada específica por chave genérica."""
        self._cache.pop(key, None)

    def clear(self) -> None:
        """Limpa todo o cache."""
        self._cache.clear()
        logger.info("Cache limpo")

    @property
    def stats(self) -> dict:
        """Estatísticas de uso do cache."""
        hit_rate = self._hits / self._total if self._total > 0 else 0.0
        return {
            "type": "memory",
            "maxsize": self._maxsize,
            "ttl_seconds": self._ttl,
            "current_size": len(self._cache),
            "hits": self._hits,
            "misses": self._misses,
            "total_requests": self._total,
            "hit_rate": round(hit_rate, 4),
        }

    @property
    def name(self) -> str:
        return f"cache_{self._maxsize}_{self._ttl}"

    @property
    def is_empty(self) -> bool:
        return len(self._cache) == 0


def cached(key_builder: Callable) -> Callable:
    """
    Decorator para cache automático em métodos de serviço.

    Uso:
        class MeuServico:
            def __init__(self):
                self.cache = PredictionCache()

            @cached(lambda self, arg1, arg2: self.cache.make_key(arg1, arg2))
            def predict(self, features, model_name):
                # computação cara...
                return resultado

    O decorator:
    1. Constrói a chave via key_builder
    2. Verifica o cache (retorna se hit)
    3. Executa a função original
    4. Armazena no cache antes de retornar
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Constrói chave
            key = key_builder(self, *args, **kwargs)
            # Verifica cache
            cached_value = self.cache.get_by_key(key)
            if cached_value is not None:
                logger.debug("@cached HIT: %s", key[:16])
                return cached_value
            # Executa função original
            result = func(self, *args, **kwargs)
            # Armazena no cache
            self.cache.set_by_key(key, result)
            return result
        return wrapper
    return decorator
