# Save this as verify_anuj.py and run it on Anuj's laptop: python verify_anuj.py
import requests
import json
import sys

# Ensure UTF-8 output encoding for Windows PowerShell terminal compatibility
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Change this to whatever Anuj's local API URL is
API_URL = "http://localhost:8000/predict" 
TEST_IMAGE_PATH = "tests/sample_test_retina.jpg" # Make sure this file exists

def verify_anuj_system():
    print(f"📡 Sending request to Anuj's API: {API_URL}...")
    
    try:
        with open(TEST_IMAGE_PATH, 'rb') as f:
            files = {'file': (TEST_IMAGE_PATH, f, 'image/jpeg')}
            response = requests.post(API_URL, files=files)
    except Exception as e:
        print(f"❌ FAIL: Could not connect to API. Is Anuj's server running?\nError: {e}")
        sys.exit(1)

    if response.status_code != 200:
        print(f"❌ FAIL: API returned error code {response.status_code}")
        print(f"Details: {response.text}")
        sys.exit(1)

    data = response.json()
    print("\n✅ API responded successfully. Checking JSON Contract...\n")
    print(json.dumps(data, indent=2))
    print("\n" + "="*50)

    # 1. Check VINAYAK's required fields
    vinayak_fields = ["grade", "probabilities", "gradcam_path"]
    for field in vinayak_fields:
        if field not in data:
            print(f"❌ FAIL: Missing VINAYAK field: '{field}'")
        else:
            print(f"✅ PASS: Found VINAYAK field: '{field}'")

    # 2. Check ANUJ's required reliability fields
    anuj_fields = ["quality", "confidence", "uncertainty", "ood", "action", "priority"]
    for field in anuj_fields:
        if field not in data:
            print(f"❌ FAIL: Missing ANUJ reliability field: '{field}'. Anuj needs to add this!")
        else:
            print(f"✅ PASS: Found ANUJ field: '{field}'")

    # 3. Final Verdict
    all_fields = vinayak_fields + anuj_fields
    if all(f in data for f in all_fields):
        print("="*50)
        print("🎉 SUCCESS! Anuj's API meets all strict integration requirements!")
        print("="*50)
    else:
        print("="*50)
        print("⚠️ WARNING: Integration is incomplete. Anuj needs to fix the missing fields above.")
        print("="*50)

if __name__ == "__main__":
    verify_anuj_system()
