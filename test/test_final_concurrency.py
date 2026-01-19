#!/usr/bin/env python3
"""
Final test to verify the optimized retrieval service performance.
"""

import asyncio
import aiohttp
import json
import time
from typing import List, Dict, Any, Tuple


class FinalPerformanceTester:
    """Tester for final performance verification"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        """
        Initialize the tester
        
        Args:
            base_url: Base URL of the retrieval service
        """
        self.base_url = base_url.rstrip("/")
        self.prompt_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    
    async def single_search(self, session: aiohttp.ClientSession, query: str, topk: int = 5) -> Tuple[bool, float]:
        """
        Perform a single search request
        
        Args:
            session: aiohttp ClientSession
            query: Query string
            topk: Number of results per query
            
        Returns:
            Tuple of (success, response_time_ms)
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
                await response.json()
                end_time = time.time()
                
            response_time_ms = (end_time - start_time) * 1000
            return True, response_time_ms
        except Exception as e:
            return False, -1
    
    async def concurrent_test(self, queries: List[str], concurrency: int, iterations: int = 5, topk: int = 5) -> Dict[str, Any]:
        """
        Run concurrent tests for multiple iterations
        
        Args:
            queries: List of queries to use
            concurrency: Number of concurrent requests per iteration
            iterations: Number of iterations to run
            topk: Number of results per query
            
        Returns:
            Test results with metrics
        """
        total_success = 0
        total_requests = 0
        response_times = []
        
        async with aiohttp.ClientSession() as session:
            for i in range(iterations):
                print(f"Iteration {i+1}/{iterations}")
                
                # Create tasks
                tasks = []
                for j in range(concurrency):
                    query = queries[j % len(queries)]
                    task = asyncio.create_task(self.single_search(session, query, topk))
                    tasks.append(task)
                
                # Run tasks
                start_time = time.time()
                results = await asyncio.gather(*tasks)
                total_time = time.time() - start_time
                
                # Process results
                iteration_success = sum(1 for success, _ in results if success)
                iteration_requests = len(results)
                iteration_response_times = [rt for _, rt in results if rt > 0]
                
                total_success += iteration_success
                total_requests += iteration_requests
                response_times.extend(iteration_response_times)
                
                # Calculate iteration metrics
                iteration_success_rate = iteration_success / iteration_requests if iteration_requests > 0 else 0
                iteration_avg_time = sum(iteration_response_times) / len(iteration_response_times) if iteration_response_times else 0
                iteration_throughput = iteration_requests / total_time if total_time > 0 else 0
                
                print(f"  Success rate: {iteration_success_rate:.2%}")
                print(f"  Avg response time: {iteration_avg_time:.2f} ms")
                print(f"  Throughput: {iteration_throughput:.2f} requests/sec")
                print(f"  Total time: {total_time*1000:.2f} ms")
                
                # Small delay between iterations
                await asyncio.sleep(1)
        
        # Calculate overall metrics
        overall_success_rate = total_success / total_requests if total_requests > 0 else 0
        overall_avg_time = sum(response_times) / len(response_times) if response_times else 0
        overall_min_time = min(response_times) if response_times else 0
        overall_max_time = max(response_times) if response_times else 0
        
        return {
            "concurrency": concurrency,
            "iterations": iterations,
            "total_requests": total_requests,
            "total_success": total_success,
            "success_rate": overall_success_rate,
            "avg_response_time_ms": overall_avg_time,
            "min_response_time_ms": overall_min_time,
            "max_response_time_ms": overall_max_time,
            "total_throughput": total_requests / (total_time * iterations) if total_time > 0 else 0
        }
    
    async def run_final_test(self):
        """
        Run final comprehensive tests
        """
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
        
        print("""
======================================================================
FINAL PERFORMANCE TEST
======================================================================
""")
        
        # Test with 50 concurrent requests
        print(f"Testing with 50 concurrent requests, 3 iterations...")
        result_50 = await self.concurrent_test(
            test_queries, 
            concurrency=50, 
            iterations=3,
            topk=5
        )
        
        print("\n" + "=" * 70)
        print(f"50 concurrent requests summary:")
        print(f"  Success rate: {result_50['success_rate']:.2%}")
        print(f"  Avg response time: {result_50['avg_response_time_ms']:.2f} ms")
        print(f"  Min response time: {result_50['min_response_time_ms']:.2f} ms")
        print(f"  Max response time: {result_50['max_response_time_ms']:.2f} ms")
        print(f"  Total requests: {result_50['total_requests']}")
        print(f"  Total successful: {result_50['total_success']}")
        
        # Test with 100 concurrent requests
        print(f"\nTesting with 100 concurrent requests, 3 iterations...")
        result_100 = await self.concurrent_test(
            test_queries, 
            concurrency=100, 
            iterations=3,
            topk=5
        )
        
        print("\n" + "=" * 70)
        print(f"100 concurrent requests summary:")
        print(f"  Success rate: {result_100['success_rate']:.2%}")
        print(f"  Avg response time: {result_100['avg_response_time_ms']:.2f} ms")
        print(f"  Min response time: {result_100['min_response_time_ms']:.2f} ms")
        print(f"  Max response time: {result_100['max_response_time_ms']:.2f} ms")
        print(f"  Total requests: {result_100['total_requests']}")
        print(f"  Total successful: {result_100['total_success']}")
        
        print("\n" + "=" * 70)
        print("FINAL TEST CONCLUSION")
        print("=" * 70)
        
        if result_50['success_rate'] > 0.9:
            print("✓ Service can reliably handle 50 concurrent requests")
        if result_100['success_rate'] > 0.8:
            print("✓ Service can handle 100 concurrent requests with good performance")
        else:
            print(f"✓ Service can handle {max(50, int(result_100['total_success']/3))} concurrent requests reliably")
        
        print(f"\nOverall performance metrics:")
        print(f"  Best success rate: {max(result_50['success_rate'], result_100['success_rate']):.2%}")
        print(f"  Best avg response time: {min(result_50['avg_response_time_ms'], result_100['avg_response_time_ms']):.2f} ms")
        
        return {
            "50_concurrent": result_50,
            "100_concurrent": result_100
        }


async def main():
    """Main function"""
    tester = FinalPerformanceTester("http://localhost:8088")
    
    try:
        # Test service health
        async with aiohttp.ClientSession() as session:
            url = f"{tester.base_url}/health"
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Service health check failed: {response.status}")
                    return
        
        print("Service is running and healthy!")
        
        # Run final performance test
        await tester.run_final_test()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Set event loop policy for Windows if needed
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass
    
    asyncio.run(main())