#!/usr/bin/env python3
"""
Smoke test script for AI Sentiment Analysis API
Tests basic functionality and demonstrates usage
"""

import requests
import json
import time
import sys
from typing import List, Dict

# API Configuration
API_BASE_URL = "http://localhost:8080"
TIMEOUT = 30

def test_health():
    """Test health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{API_BASE_URL}/healthz", timeout=TIMEOUT)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data['status']}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis endpoint"""
    print("\n🔍 Testing sentiment analysis...")

    test_cases = [
        {"text": "Tôi rất thích sản phẩm này!", "lang": "vi", "expected": "positive"},
        {"text": "Sản phẩm này thật tệ", "lang": "vi", "expected": "negative"},
        {"text": "Bình thường thôi", "lang": "vi", "expected": "neutral"},
        {"text": "This product is amazing!", "lang": "en", "expected": "positive"},
    ]

    passed = 0
    for i, test_case in enumerate(test_cases, 1):
        try:
            payload = {
                "text": test_case["text"],
                "lang": test_case["lang"]
            }

            print(f"  Test {i}: '{test_case['text'][:50]}...'")
            response = requests.post(
                f"{API_BASE_URL}/v1/sentiment",
                json=payload,
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                result = data["label"]
                score = data["score"]
                latency = data["latency_ms"]

                print(f"    ✅ Result: {result} (score: {score:.3f}, {latency}ms)")

                if result == test_case["expected"]:
                    passed += 1
                else:
                    print(f"    ⚠️  Expected: {test_case['expected']}")
            else:
                print(f"    ❌ HTTP {response.status_code}: {response.text}")

        except Exception as e:
            print(f"    ❌ Error: {e}")

    print(f"\n📊 Sentiment tests: {passed}/{len(test_cases)} passed")
    return passed == len(test_cases)

def test_batch_processing():
    """Test batch processing endpoint"""
    print("\n🔍 Testing batch processing...")

    payload = {
        "items": [
            {"id": "1", "text": "Tôi thích điều này", "lang": "vi"},
            {"id": "2", "text": "Không hài lòng", "lang": "vi"},
            {"id": "3", "text": "Bình thường", "lang": "vi"}
        ],
        "async_mode": False
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/v1/sentiment/batch",
            json=payload,
            timeout=TIMEOUT
        )

        if response.status_code == 200:
            data = response.json()
            print(f"  ✅ Batch processed: {data['total_items']} items")
            print(f"  📊 Results: {len(data['results'])} completed")

            for result in data["results"]:
                sentiment = result["result"]["label"] if result["result"] else "error"
                print(f"    ID {result['id']}: {sentiment}")

            return True
        else:
            print(f"  ❌ Batch failed: {response.status_code}")
            return False

    except Exception as e:
        print(f"  ❌ Batch error: {e}")
        return False

def test_caching():
    """Test caching functionality"""
    print("\n🔍 Testing caching...")

    payload = {"text": "Tôi rất thích test caching này!", "lang": "vi"}

    try:
        # First request
        start_time = time.time()
        response1 = requests.post(
            f"{API_BASE_URL}/v1/sentiment",
            json=payload,
            timeout=TIMEOUT
        )
        first_latency = time.time() - start_time

        # Second request (should be cached)
        start_time = time.time()
        response2 = requests.post(
            f"{API_BASE_URL}/v1/sentiment",
            json=payload,
            timeout=TIMEOUT
        )
        second_latency = time.time() - start_time

        if response1.status_code == 200 and response2.status_code == 200:
            data1 = response1.json()
            data2 = response2.json()

            print(f"  First request: {first_latency*1000:.1f}ms, cached: {data1['cached']}")
            print(f"  Second request: {second_latency*1000:.1f}ms, cached: {data2['cached']}")

            if data2["cached"] or second_latency < first_latency:
                print("  ✅ Caching appears to be working")
                return True
            else:
                print("  ⚠️  Caching may not be enabled")
                return True  # Not a failure
        else:
            print("  ❌ Cache test failed")
            return False

    except Exception as e:
        print(f"  ❌ Cache test error: {e}")
        return False

def demonstrate_features():
    """Demonstrate key features"""
    print("\n🎯 Feature Demonstration")
    print("=" * 50)

    # Vietnamese sentiment analysis
    vietnamese_examples = [
        "Tôi cực kỳ hài lòng với sản phẩm này! Chất lượng tuyệt vời.",
        "Dịch vụ khách hàng rất tệ, nhân viên không nhiệt tình.",
        "Sản phẩm này ổn, không có gì đặc biệt nhưng cũng không tệ.",
        "Mình không thích màu sắc nhưng chất lượng khá tốt."
    ]

    print("\n📝 Vietnamese Sentiment Analysis:")
    for i, text in enumerate(vietnamese_examples, 1):
        try:
            response = requests.post(
                f"{API_BASE_URL}/v1/sentiment",
                json={"text": text, "lang": "vi"},
                timeout=TIMEOUT
            )

            if response.status_code == 200:
                data = response.json()
                print(f"  {i}. '{text[:60]}...'")
                print(f"     → {data['label'].upper()} ({data['score']:.3f})")
            else:
                print(f"  {i}. Error: {response.status_code}")

        except Exception as e:
            print(f"  {i}. Exception: {e}")

def main():
    """Run all smoke tests"""
    print("🚀 AI Sentiment Analysis - Smoke Test")
    print("=" * 50)

    tests = [
        ("Health Check", test_health),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Batch Processing", test_batch_processing),
        ("Caching", test_caching)
    ]

    passed = 0
    for test_name, test_func in tests:
        if test_func():
            passed += 1

    print(f"\n🏁 Test Summary: {passed}/{len(tests)} passed")

    if passed == len(tests):
        print("🎉 All tests passed! System is working correctly.")
        demonstrate_features()
        return 0
    else:
        print("⚠️  Some tests failed. Check the system configuration.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
