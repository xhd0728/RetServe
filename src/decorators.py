"""
Decorator utilities for cross-cutting concerns.

This module provides decorators for common patterns like logging,
timing, retrying, and input validation. These decorators help
maintain clean separation of concerns in the codebase.
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from typing import Any, Callable, ParamSpec, TypeVar

# Type variables for generic decorators
P = ParamSpec("P")
T = TypeVar("T")

# Module logger
logger = logging.getLogger(__name__)


# =============================================================================
# Logging Decorators
# =============================================================================

def log_execution(
    level: int = logging.INFO,
    log_args: bool = False,
    log_result: bool = False,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to log function execution start and completion.
    
    Args:
        level: Logging level (default: INFO).
        log_args: Whether to log function arguments.
        log_result: Whether to log function result.
        
    Returns:
        Decorated function.
        
    Example:
        @log_execution(log_args=True)
        async def my_function(x, y):
            return x + y
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__qualname__
            
            # Log start
            if log_args:
                logger.log(level, f"Executing {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"Executing {func_name}")
            
            try:
                result = await func(*args, **kwargs)
                
                # Log completion
                if log_result:
                    logger.log(level, f"Completed {func_name} with result={result}")
                else:
                    logger.log(level, f"Completed {func_name}")
                
                return result
            except Exception as exc:
                logger.exception(f"Error in {func_name}: {exc}")
                raise
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            func_name = func.__qualname__
            
            # Log start
            if log_args:
                logger.log(level, f"Executing {func_name} with args={args}, kwargs={kwargs}")
            else:
                logger.log(level, f"Executing {func_name}")
            
            try:
                result = func(*args, **kwargs)
                
                # Log completion
                if log_result:
                    logger.log(level, f"Completed {func_name} with result={result}")
                else:
                    logger.log(level, f"Completed {func_name}")
                
                return result
            except Exception as exc:
                logger.exception(f"Error in {func_name}: {exc}")
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Timing Decorators
# =============================================================================

def measure_time(
    log_level: int = logging.DEBUG,
    threshold_ms: float | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to measure and log function execution time.
    
    Args:
        log_level: Logging level for timing messages.
        threshold_ms: Only log if execution time exceeds this threshold (ms).
        
    Returns:
        Decorated function with timing instrumentation.
        
    Example:
        @measure_time(threshold_ms=100)
        async def slow_operation():
            await asyncio.sleep(0.2)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            try:
                return await func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if threshold_ms is None or elapsed_ms >= threshold_ms:
                    logger.log(
                        log_level,
                        f"{func.__qualname__} executed in {elapsed_ms:.2f}ms"
                    )
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            start_time = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                if threshold_ms is None or elapsed_ms >= threshold_ms:
                    logger.log(
                        log_level,
                        f"{func.__qualname__} executed in {elapsed_ms:.2f}ms"
                    )
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Retry Decorators
# =============================================================================

def retry(
    max_attempts: int = 3,
    delay_seconds: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[Exception, int], None] | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to retry function execution on failure with exponential backoff.
    
    Args:
        max_attempts: Maximum number of retry attempts.
        delay_seconds: Initial delay between retries in seconds.
        backoff_factor: Multiplier for delay on each retry.
        exceptions: Tuple of exception types to catch and retry.
        on_retry: Optional callback called on each retry with (exception, attempt).
        
    Returns:
        Decorated function with retry logic.
        
    Example:
        @retry(max_attempts=3, delay_seconds=1.0)
        async def unreliable_api_call():
            response = await client.get("/api/data")
            return response.json()
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            current_delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__qualname__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__qualname__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(exc, attempt)
                    
                    await asyncio.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception  # type: ignore
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            last_exception: Exception | None = None
            current_delay = delay_seconds
            
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exception = exc
                    
                    if attempt == max_attempts:
                        logger.error(
                            f"{func.__qualname__} failed after {max_attempts} attempts: {exc}"
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__qualname__} attempt {attempt}/{max_attempts} failed: {exc}. "
                        f"Retrying in {current_delay:.1f}s..."
                    )
                    
                    if on_retry:
                        on_retry(exc, attempt)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
            
            # This should never be reached, but just in case
            raise last_exception  # type: ignore
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Validation Decorators
# =============================================================================

def validate_input(
    validator: Callable[..., bool],
    error_message: str = "Input validation failed",
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to validate function inputs before execution.
    
    Args:
        validator: Function that takes the same arguments and returns True if valid.
        error_message: Error message to raise if validation fails.
        
    Returns:
        Decorated function with input validation.
        
    Example:
        def validate_positive(x: int) -> bool:
            return x > 0
            
        @validate_input(validate_positive, "x must be positive")
        def process(x: int) -> int:
            return x * 2
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not validator(*args, **kwargs):
                raise ValueError(error_message)
            return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            if not validator(*args, **kwargs):
                raise ValueError(error_message)
            return func(*args, **kwargs)
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return sync_wrapper  # type: ignore
    
    return decorator


# =============================================================================
# Concurrency Control Decorators
# =============================================================================

def with_semaphore(
    semaphore: asyncio.Semaphore,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to limit concurrent execution using a semaphore.
    
    Args:
        semaphore: asyncio.Semaphore for concurrency control.
        
    Returns:
        Decorated function with semaphore protection.
        
    Example:
        embedding_semaphore = asyncio.Semaphore(32)
        
        @with_semaphore(embedding_semaphore)
        async def get_embeddings(texts: list[str]):
            return await client.embed(texts)
    """
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            async with semaphore:
                return await func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator


def singleton(cls: type[T]) -> type[T]:
    """
    Decorator to make a class a singleton.
    
    Args:
        cls: The class to make a singleton.
        
    Returns:
        The singleton class.
        
    Example:
        @singleton
        class ConfigManager:
            def __init__(self):
                self.config = {}
    """
    instances: dict[type, Any] = {}
    
    @functools.wraps(cls)
    def get_instance(*args: Any, **kwargs: Any) -> T:
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    
    return get_instance  # type: ignore
