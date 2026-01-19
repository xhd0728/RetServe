#!/usr/bin/env python3
"""
Test script for evaluating retrieval service concurrent performance.

This script tests the service's ability to handle multiple concurrent requests and measures response times.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict, Any, Tuple


class ConcurrentRetrievalTester:
    """Tester for concurrent retrieval service requests"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        """
        Initialize the tester
        
        Args:
            base_url: Base URL of the retrieval service
        """
        self.base_url = base_url.rstrip("/")
        self.prompt_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    
    async def health_check(self, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """
        Check the health of the service
        
        Args:
            session: aiohttp ClientSession
            
        Returns:
            Health check response
        """
        url = f"{self.base_url}/health"
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    
    async def single_search(self, session: aiohttp.ClientSession, query: str, topk: int = 5) -> Tuple[float, Dict[str, Any]]:
        """
        Perform a single search request
        
        Args:
            session: aiohttp ClientSession
            query: Query string
            topk: Number of results per query
            
        Returns:
            Tuple of (response_time_ms, result)
        """
        # Prepend prompt to query
        formatted_query = f"{self.prompt_prefix}{query}"
        
        url = f"{self.base_url}/search"
        payload = {
            "queries": [formatted_query],
            "topk": topk
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        start_time = time.time()
        async with session.post(url, data=json.dumps(payload), headers=headers) as response:
            response.raise_for_status()
            result = await response.json()
            end_time = time.time()
            
        response_time_ms = (end_time - start_time) * 1000
        return response_time_ms, result
    
    async def concurrent_search(self, queries: List[str], concurrency: int, topk: int = 5) -> Dict[str, Any]:
        """
        Perform concurrent search requests
        
        Args:
            queries: List of query strings to distribute among requests
            concurrency: Number of concurrent requests
            topk: Number of results per query
            
        Returns:
            Performance metrics and results
        """
        # Create a session with connection pooling
        connector = aiohttp.TCPConnector(limit=concurrency)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Create tasks
            tasks = []
            for i in range(concurrency):
                # Cycle through queries
                query = queries[i % len(queries)]
                task = asyncio.create_task(self.single_search(session, query, topk))
                tasks.append(task)
            
            # Run tasks
            start_time = time.time()
            results = await asyncio.gather(*tasks, return_exceptions=True)
            total_time_ms = (time.time() - start_time) * 1000
        
        # Process results
        successful_requests = []
        failed_requests = []
        response_times = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_requests.append({
                    "request_id": i,
                    "error": str(result)
                })
            else:
                response_time_ms, data = result
                successful_requests.append(data)
                response_times.append(response_time_ms)
        
        # Calculate metrics
        metrics = {
            "total_requests": concurrency,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / concurrency if concurrency > 0 else 0,
            "total_time_ms": total_time_ms,
            "avg_time_per_request_ms": statistics.mean(response_times) if response_times else 0,
            "min_time_ms": min(response_times) if response_times else 0,
            "max_time_ms": max(response_times) if response_times else 0,
            "p95_time_ms": sorted(response_times)[int(len(response_times) * 0.95)] if len(response_times) > 20 else 0,
            "throughput": concurrency / (total_time_ms / 1000) if total_time_ms > 0 else 0
        }
        
        return {
            "metrics": metrics,
            "failed_requests": failed_requests
        }
    
    async def load_test(self, queries: List[str], concurrency_levels: List[int], topk: int = 5) -> Dict[str, Any]:
        """
        Run load tests with different concurrency levels
        
        Args:
            queries: List of queries to use
            concurrency_levels: List of concurrency levels to test
            topk: Number of results per query
            
        Returns:
            Load test results
        """
        load_test_results = []
        
        for concurrency in sorted(concurrency_levels):
            print(f"Testing {concurrency} concurrent requests...")
            
            # Run test with current concurrency level
            result = await self.concurrent_search(queries, concurrency, topk)
            load_test_results.append({
                "concurrency": concurrency,
                **result
            })
            
            # Print intermediate results
            metrics = result["metrics"]
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            print(f"  Avg response time: {metrics['avg_time_per_request_ms']:.2f} ms")
            print(f"  Throughput: {metrics['throughput']:.2f} requests/sec")
            print(f"  Total time: {metrics['total_time_ms']:.2f} ms")
            
            # Small delay between tests
            await asyncio.sleep(1)
        
        return {
            "test_config": {
                "queries": queries,
                "concurrency_levels": concurrency_levels,
                "topk": topk
            },
            "results": load_test_results
        }


def print_load_test_summary(load_test_results: Dict[str, Any]) -> None:
    """
    Print a summary of the load test results
    
    Args:
        load_test_results: Results from the load test
    """
    print("\n" + "=" * 70)
    print("LOAD TEST SUMMARY")
    print("=" * 70)
    
    print(f"Tested concurrency levels: {load_test_results['test_config']['concurrency_levels']}")
    print(f"Topk per query: {load_test_results['test_config']['topk']}")
    print(f"Queries used: {load_test_results['test_config']['queries']}")
    
    print("\n" + "=" * 70)
    print(f"{'Concurrency':<12} {'Success Rate':<12} {'Avg Time (ms)':<15} {'Throughput':<15} {'Total Time (ms)':<15}")
    print("-" * 70)
    
    for result in load_test_results['results']:
        metrics = result['metrics']
        print(f"{metrics['total_requests']:<12} {metrics['success_rate']:<11.2%} {metrics['avg_time_per_request_ms']:<14.2f} {metrics['throughput']:<14.2f} {metrics['total_time_ms']:<14.2f}")
    
    # Find best throughput
    best_throughput = max(result['metrics']['throughput'] for result in load_test_results['results'])
    best_result = next(r for r in load_test_results['results'] if r['metrics']['throughput'] == best_throughput)
    
    print("\n" + "=" * 70)
    print(f"Best throughput: {best_throughput:.2f} requests/sec at concurrency {best_result['concurrency']}")
    
    # Report failures if any
    total_failures = sum(len(r['failed_requests']) for r in load_test_results['results'])
    if total_failures > 0:
        print(f"\nTotal failed requests: {total_failures}")
        for result in load_test_results['results']:
            if result['failed_requests']:
                print(f"  Concurrency {result['concurrency']}: {len(result['failed_requests'])} failures")
                for failure in result['failed_requests'][:3]:  # Show first 3 failures
                    print(f"    - Request {failure['request_id']}: {failure['error'][:100]}...")
                if len(result['failed_requests']) > 3:
                    print(f"    - ... and {len(result['failed_requests']) - 3} more")


async def main():
    """Main function"""
    tester = ConcurrentRetrievalTester("http://localhost:8088")
    
    # Test queries
    test_queries = [
        "三国时期刘备的主要事迹",
        "诸葛亮草船借箭的故事",
        "赤壁之战的背景和结果",
        "关羽过五关斩六将的故事",
        "曹操煮酒论英雄的典故",
        "诸葛亮空城计的背景",
        "周瑜打黄盖的苦肉计",
        "赵云单骑救主的故事",
        "桃园三结义的人物",
        "七擒孟获的过程"
    ]
    
    try:
        # Create session for health check
        async with aiohttp.ClientSession() as session:
            print("Checking service health...")
            health = await tester.health_check(session)
            print(f"Service status: {health['status']}")
            print(f"Corpus size: {health['corpus_size']}")
            print(f"Index dimension: {health['index_dim']}")
        
        # Run load tests with different concurrency levels
        concurrency_levels = [10, 20, 40, 80, 120, 160]
        topk = 5
        
        print("\n" + "=" * 70)
        print(f"Running load tests with topk={topk}")
        print("=" * 70)
        
        load_test_results = await tester.load_test(test_queries, concurrency_levels, topk)
        
        # Print comprehensive summary
        print_load_test_summary(load_test_results)
        
    except aiohttp.ClientError as e:
        print(f"Error: Could not connect to the retrieval service: {e}")
        print("Make sure the service is running with: python ret_serve.py")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set event loop policy for Windows if needed
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass
    
    asyncio.run(main())