"""Benchmark JSON deserialization performance.

Run with: pytest tests/benches/test_json_deserialization.py -v -s
"""

import json
import time
from typing import Any

import numpy as np
import pytest

from qdrant_client import _json


def generate_search_response(num_results: int, vector_size: int) -> bytes:
    """Generate a realistic search response payload."""
    results = []
    for i in range(num_results):
        results.append(
            {
                "id": i,
                "version": 1,
                "score": float(np.random.rand()),
                "vector": np.random.rand(vector_size).tolist(),
                "payload": {
                    "text": f"This is document {i} with some text content that might be typical.",
                    "category": f"category_{i % 10}",
                    "tags": [f"tag_{j}" for j in range(5)],
                    "metadata": {
                        "created_at": "2024-01-15T10:30:00Z",
                        "updated_at": "2024-01-16T14:20:00Z",
                        "author": f"author_{i % 100}",
                        "views": i * 10,
                    },
                },
            }
        )
    response = {"result": results, "status": "ok", "time": 0.001}
    return json.dumps(response).encode("utf-8")


def generate_scroll_response(num_points: int, vector_size: int) -> bytes:
    """Generate a realistic scroll response payload."""
    points = []
    for i in range(num_points):
        points.append(
            {
                "id": i,
                "vector": np.random.rand(vector_size).tolist(),
                "payload": {
                    "title": f"Document title {i}",
                    "content": f"Lorem ipsum dolor sit amet, consectetur adipiscing elit. " * 5,
                    "embedding_model": "text-embedding-3-small",
                    "chunk_index": i % 50,
                    "source_url": f"https://example.com/docs/{i}",
                },
            }
        )
    response = {
        "result": {"points": points, "next_page_offset": num_points},
        "status": "ok",
        "time": 0.002,
    }
    return json.dumps(response).encode("utf-8")


def benchmark_loads(data: bytes, iterations: int) -> tuple[float, Any]:
    """Benchmark _json.loads and return (total_time, result)."""
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = _json.loads(data)
    end = time.perf_counter()
    return end - start, result


def benchmark_stdlib_loads(data: bytes, iterations: int) -> tuple[float, Any]:
    """Benchmark stdlib json.loads for comparison."""
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = json.loads(data)
    end = time.perf_counter()
    return end - start, result


@pytest.mark.skip(reason="skip slow benchmark")
def test_json_backend_info():
    """Print which JSON backend is being used."""
    print(f"\nJSON backend: {_json.BACKEND}")


@pytest.mark.skip(reason="skip slow benchmark")
def test_search_response_deserialization():
    """Benchmark deserializing search responses."""
    print(f"\nJSON backend: {_json.BACKEND}")
    print("\n--- Search Response Deserialization ---")

    configs = [
        (10, 384, 1000),  # Small: 10 results, 384-dim vectors
        (100, 384, 100),  # Medium: 100 results, 384-dim vectors
        (100, 1536, 100),  # Large vectors: 100 results, 1536-dim vectors
    ]

    for num_results, vector_size, iterations in configs:
        data = generate_search_response(num_results, vector_size)
        size_kb = len(data) / 1024

        time_current, _ = benchmark_loads(data, iterations)
        time_stdlib, _ = benchmark_stdlib_loads(data, iterations)

        print(f"\n{num_results} results, {vector_size}-dim vectors ({size_kb:.1f} KB):")
        print(f"  _json.loads:  {time_current:.4f}s ({iterations} iterations)")
        print(f"  stdlib json:  {time_stdlib:.4f}s ({iterations} iterations)")
        print(f"  Speedup:      {time_stdlib / time_current:.2f}x")


@pytest.mark.skip(reason="skip slow benchmark")
def test_scroll_response_deserialization():
    """Benchmark deserializing scroll responses."""
    print(f"\nJSON backend: {_json.BACKEND}")
    print("\n--- Scroll Response Deserialization ---")

    configs = [
        (100, 384, 100),  # 100 points, 384-dim
        (500, 384, 50),  # 500 points, 384-dim
        (1000, 768, 20),  # 1000 points, 768-dim
    ]

    for num_points, vector_size, iterations in configs:
        data = generate_scroll_response(num_points, vector_size)
        size_kb = len(data) / 1024

        time_current, _ = benchmark_loads(data, iterations)
        time_stdlib, _ = benchmark_stdlib_loads(data, iterations)

        print(f"\n{num_points} points, {vector_size}-dim vectors ({size_kb:.1f} KB):")
        print(f"  _json.loads:  {time_current:.4f}s ({iterations} iterations)")
        print(f"  stdlib json:  {time_stdlib:.4f}s ({iterations} iterations)")
        print(f"  Speedup:      {time_stdlib / time_current:.2f}x")


@pytest.mark.skip(reason="skip slow benchmark")
def test_large_payload_deserialization():
    """Benchmark with large payloads typical of RAG applications."""
    print(f"\nJSON backend: {_json.BACKEND}")
    print("\n--- Large Payload Deserialization ---")

    # Simulate a large retrieval response with text chunks
    num_chunks = 50
    chunk_size = 2000  # characters per chunk
    vector_size = 1536

    points = []
    for i in range(num_chunks):
        points.append(
            {
                "id": f"doc_{i}",
                "vector": np.random.rand(vector_size).tolist(),
                "payload": {
                    "text": "x" * chunk_size,
                    "source": f"document_{i // 10}.pdf",
                    "page": i % 20,
                },
            }
        )
    response = {"result": points, "status": "ok", "time": 0.005}
    data = json.dumps(response).encode("utf-8")
    size_mb = len(data) / (1024 * 1024)

    iterations = 50
    time_current, _ = benchmark_loads(data, iterations)
    time_stdlib, _ = benchmark_stdlib_loads(data, iterations)

    print(
        f"\n{num_chunks} chunks with {chunk_size}-char text, {vector_size}-dim ({size_mb:.2f} MB):"
    )
    print(f"  _json.loads:  {time_current:.4f}s ({iterations} iterations)")
    print(f"  stdlib json:  {time_stdlib:.4f}s ({iterations} iterations)")
    print(f"  Speedup:      {time_stdlib / time_current:.2f}x")
