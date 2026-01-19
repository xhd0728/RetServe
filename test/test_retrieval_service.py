#!/usr/bin/env python3
"""
Test script for the retrieval service.

This script demonstrates how to call the retrieval service API with the required prompt format.
"""

import requests
import json
from typing import List, Dict, Any, Optional


class RetrievalClient:
    """Client for the retrieval service"""
    
    def __init__(self, base_url: str = "http://localhost:8088"):
        """
        Initialize the client
        
        Args:
            base_url: Base URL of the retrieval service
        """
        self.base_url = base_url.rstrip("/")
        self.prompt_prefix = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the service
        
        Returns:
            Health check response
        """
        url = f"{self.base_url}/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def search(self, queries: List[str], topk: int = 5) -> Dict[str, Any]:
        """
        Search the retrieval service with the given queries
        
        Args:
            queries: List of query strings
            topk: Number of results per query
            
        Returns:
            Search results
        """
        # Prepend prompt to each query
        formatted_queries = [f"{self.prompt_prefix}{query}" for query in queries]
        
        url = f"{self.base_url}/search"
        payload = {
            "queries": formatted_queries,
            "topk": topk
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        response.raise_for_status()
        return response.json()
    
    def search_single(self, query: str, topk: int = 5) -> Dict[str, Any]:
        """
        Search with a single query
        
        Args:
            query: Query string
            topk: Number of results
            
        Returns:
            Search results for the single query
        """
        result = self.search([query], topk)
        # Extract results for the single query
        if result["contents"] and result["scores"]:
            return {
                "contents": result["contents"][0],
                "scores": result["scores"][0]
            }
        return {"contents": [], "scores": []}


def print_search_results(query: str, results: Dict[str, Any]) -> None:
    """
    Print search results in a readable format
    
    Args:
        query: Original query
        results: Search results
    """
    print(f"\nQuery: {query}")
    print("=" * 50)
    
    if not results["contents"]:
        print("No results found")
        return
    
    for i, (content, score) in enumerate(zip(results["contents"], results["scores"])):
        print(f"\nResult {i+1} (Score: {score:.4f})")
        print(f"ID: {content.get('id', 'N/A')}")
        print(f"Title: {content.get('title', 'N/A')}")
        print(f"Text: {content.get('text', 'N/A')[:200]}...")
    
    print("\n" + "=" * 50)


def main():
    """Main function"""
    # Initialize client
    client = RetrievalClient("http://localhost:8088")
    
    try:
        # Check service health
        print("Checking service health...")
        health = client.health_check()
        print(f"Service status: {health['status']}")
        print(f"Corpus size: {health['corpus_size']}")
        print(f"Index dimension: {health['index_dim']}")
        
        # Example queries
        test_queries = [
            "三国时期刘备的主要事迹",
            "诸葛亮草船借箭的故事",
            "赤壁之战的背景和结果"
        ]
        
        # Perform searches
        print("\n" + "=" * 60)
        print("Performing test searches...")
        print("=" * 60)
        
        for query in test_queries:
            results = client.search_single(query, topk=3)
            print_search_results(query, results)
            
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the retrieval service.")
        print("Make sure the service is running with: python ret_serve.py")
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error occurred: {e}")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()