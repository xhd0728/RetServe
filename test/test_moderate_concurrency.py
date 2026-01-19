#!/usr/bin/env python3
"""
Test script for evaluating retrieval service with moderate concurrent requests.

This script tests the service with reasonable concurrency levels to avoid crashing it.
"""

import asyncio
import aiohttp
import json
import time
import statistics
from typing import List, Dict, Any, Tuple


class ModerateConcurrencyTester:
    """Tester for moderate concurrent retrieval service requests"""
    
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
        try:
            async with session.post(url, data=json.dumps(payload), headers=headers) as response:
                response.raise_for_status()
                result = await response.json()
                end_time = time.time()
                
            response_time_ms = (end_time - start_time) * 1000
            return response_time_ms, result
        except Exception as e:
            return -1, {"error": str(e)}
    
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
            
            # Run tasks with timeout to prevent hanging
            start_time = time.time()
            try:
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks), 
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                return {
                    "metrics": {
                        "total_requests": concurrency,
                        "success_rate": 0.0,
                        "error": "Timeout occurred"
                    },
                    "failed_requests": ["Timeout"] * concurrency
                }
            
            total_time_ms = (time.time() - start_time) * 1000
        
        # Process results
        successful_requests = []
        failed_requests = []
        response_times = []
        
        for i, (response_time_ms, result) in enumerate(results):
            if response_time_ms < 0 or "error" in result:
                failed_requests.append({
                    "request_id": i,
                    "error": result.get("error", "Unknown error")
                })
            else:
                successful_requests.append(result)
                response_times.append(response_time_ms)
        
        # Calculate metrics
        metrics = {
            "total_requests": concurrency,
            "successful_requests": len(successful_requests),
            "failed_requests": len(failed_requests),
            "success_rate": len(successful_requests) / concurrency if concurrency > 0 else 0,
            "total_time_ms": total_time_ms,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "min_time_ms": min(response_times) if response_times else 0,
            "max_time_ms": max(response_times) if response_times else 0,
            "throughput": concurrency / (total_time_ms / 1000) if total_time_ms > 0 else 0
        }
        
        return {
            "metrics": metrics,
            "failed_requests": failed_requests
        }
    
    async def progressive_load_test(self, queries: List[str], concurrency_start: int = 5, concurrency_end: int = 30, step: int = 5) -> Dict[str, Any]:
        """
        Run progressive load tests with increasing concurrency levels
        
        Args:
            queries: List of queries to use
            concurrency_start: Starting concurrency level
            concurrency_end: Maximum concurrency level
            step: Step size between concurrency levels
            
        Returns:
            Load test results
        """
        results = []
        current_concurrency = concurrency_start
        
        while current_concurrency <= concurrency_end:
            print(f"Testing {current_concurrency} concurrent requests...")
            
            # Run test with current concurrency level
            result = await self.concurrent_search(queries, current_concurrency, topk=5)
            results.append({
                "concurrency": current_concurrency,
                **result
            })
            
            # Print intermediate results
            metrics = result["metrics"]
            print(f"  Success rate: {metrics['success_rate']:.2%}")
            if metrics['successful_requests'] > 0:
                print(f"  Avg response time: {metrics['avg_response_time_ms']:.2f} ms")
                print(f"  Throughput: {metrics['throughput']:.2f} requests/sec")
            print(f"  Total time: {metrics['total_time_ms']:.2f} ms")
            
            # Check if we're getting too many failures
            if metrics['success_rate'] < 0.5 and current_concurrency > concurrency_start:
                print(f"  Warning: Low success rate ({metrics['success_rate']:.2%}), stopping progressive tests")
                break
            
            # Small delay between tests
            await asyncio.sleep(2)
            
            # Increase concurrency for next test
            current_concurrency += step
        
        return {
            "test_config": {
                "queries": queries,
                "concurrency_start": concurrency_start,
                "concurrency_end": concurrency_end,
                "step": step
            },
            "results": results
        }


def print_test_summary(test_results: Dict[str, Any]) -> None:
    """
    Print a summary of the test results
    
    Args:
        test_results: Results from the test
    """
    print("\n" + "=" * 70)
    print("MODERATE CONCURRENCY TEST SUMMARY")
    print("=" * 70)
    
    config = test_results['test_config']
    print(f"Test range: {config['concurrency_start']} to {config['concurrency_end']} concurrent requests")
    print(f"Step size: {config['step']}")
    
    print("\n" + "=" * 70)
    print(f"{'Concurrency':<12} {'Success Rate':<12} {'Avg Time (ms)':<15} {'Throughput':<15} {'Total Time (ms)':<15}")
    print("-" * 70)
    
    for result in test_results['results']:
        metrics = result['metrics']
        print(f"{metrics['total_requests']:<12} {metrics['success_rate']:<11.2%} "
              f"{metrics['avg_response_time_ms']:<14.2f} {metrics['throughput']:<14.2f} "
              f"{metrics['total_time_ms']:<14.2f}")
    
    # Find best throughput
    successful_runs = [r for r in test_results['results'] if r['metrics']['success_rate'] > 0.5]
    if successful_runs:
        best_run = max(successful_runs, key=lambda r: r['metrics']['throughput'])
        print(f"\nBest performance: {best_run['metrics']['throughput']:.2f} requests/sec at concurrency {best_run['concurrency']}")
    
    # Report failures
    total_failures = sum(r['metrics']['failed_requests'] for r in test_results['results'])
    if total_failures > 0:
        print(f"\nTotal failed requests: {total_failures}")
        for result in test_results['results']:
            if result['metrics']['failed_requests'] > 0:
                print(f"  Concurrency {result['concurrency']}: {result['metrics']['failed_requests']} failures")


async def main():
    """Main function"""
    tester = ModerateConcurrencyTester("http://localhost:8088")
    
    # Test queries
    test_queries = [
        "三国时期刘备的主要事迹",
        "诸葛亮草船借箭的故事",
        "赤壁之战的背景和结果",
        "关羽过五关斩六将的故事",
        "曹操煮酒论英雄的典故"
    ]
    
    try:
        # Create session for health check
        async with aiohttp.ClientSession() as session:
            print("Checking service health...")
            health = await tester.health_check(session)
            print(f"Service status: {health['status']}")
            print(f"Corpus size: {health['corpus_size']}")
            print(f"Index dimension: {health['index_dim']}")
        
        # Run progressive load tests
        print("\n" + "=" * 70)
        print("Running progressive concurrency tests")
        print("=" * 70)
        
        test_results = await tester.progressive_load_test(
            test_queries, 
            concurrency_start=5, 
            concurrency_end=30, 
            step=5
        )
        
        # Print comprehensive summary
        print_test_summary(test_results)
        
        # Test the max_topk soft limit
        print("\n" + "=" * 70)
        print("Testing max_topk soft limit")
        print("=" * 70)
        
        async with aiohttp.ClientSession() as session:
            # Test with topk exceeding the configured limit
            query = "诸葛亮的主要事迹"
            formatted_query = f"{tester.prompt_prefix}{query}"
            
            url = f"{tester.base_url}/search"
            payload = {
                "queries": [formatted_query],
                "topk": 2000  # This should exceed the configured max_topk of 999
            }
            
            headers = {"Content-Type": "application/json"}
            
            async with session.post(url, data=json.dumps(payload), headers=headers) as response:
                if response.status == 200:
                    result = await response.json()
                    print(f"✓ Request with topk=2000 succeeded (soft limit working)")
                    if result.get("contents"):
                        actual_topk = len(result["contents"][0])
                        print(f"  Received {actual_topk} results (should be <= 999)")
                else:
                    print(f"✗ Request failed with status: {response.status}")
                    print(f"  Error: {await response.text()}")
            
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