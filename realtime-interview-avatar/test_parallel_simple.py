# -*- coding: utf-8 -*-
"""
Simple parallel generation test
"""

import time
import requests
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = "http://localhost:5000"

def send_request(user_id, text):
    """Send generation request"""
    print(f"[User {user_id}] Sending request...")
    start = time.time()

    try:
        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "avatar_path": "precomputed/a24ca21b-0d02-49f0-99a7-d737c5ca6058_precomputed.pkl",
                "text": text,
                "tts_engine": "edge",
                "tts_voice": "ko-KR-SunHiNeural",
                "sid": f"test_user_{user_id}",
                "resolution": "480p"
            },
            timeout=180
        )
        elapsed = time.time() - start
        print(f"[User {user_id}] Response received in {elapsed:.2f}s: {response.status_code}")
        return {"user_id": user_id, "elapsed": elapsed, "status": response.status_code}
    except Exception as e:
        elapsed = time.time() - start
        print(f"[User {user_id}] Error: {e}")
        return {"user_id": user_id, "elapsed": elapsed, "error": str(e)}

def monitor_queue():
    """Monitor queue status"""
    for _ in range(30):
        try:
            r = requests.get(f"{BASE_URL}/api/queue_status", timeout=3)
            s = r.json()
            gpu = s.get('gpu', {})
            print(f"  [Queue] processing={s.get('processing_count')}/{s.get('max_concurrent')} "
                  f"waiting={s.get('queue_length')} GPU={gpu.get('memory_used_mb', '?')}MB/{gpu.get('memory_total_mb', '?')}MB")
        except:
            pass
        time.sleep(2)

if __name__ == "__main__":
    print("="*60)
    print("Parallel Generation Test")
    print("="*60)

    # Check server
    try:
        r = requests.get(f"{BASE_URL}/api/queue_status", timeout=5)
        s = r.json()
        print(f"Server OK - GPU: {s.get('gpu', {}).get('memory_used_mb', '?')}MB used")
    except Exception as e:
        print(f"Server not available: {e}")
        exit(1)

    # Start monitor
    monitor = threading.Thread(target=monitor_queue, daemon=True)
    monitor.start()

    # Send 2 parallel requests
    print("\nSending 2 parallel requests...")
    texts = [
        "Hello. I am the first user. Thank you for joining the interview today.",
        "Hello. I am the second user. I will conduct the interview today."
    ]

    total_start = time.time()
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(send_request, i+1, texts[i]) for i in range(2)]
        time.sleep(0.3)  # Small delay between requests

        results = [f.result() for f in futures]

    total_elapsed = time.time() - total_start

    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    for r in results:
        print(f"  User {r['user_id']}: {r['elapsed']:.2f}s")
    print(f"\nTotal time: {total_elapsed:.2f}s")
    print("="*60)
