import json
import requests
import time

# Wait for server
time.sleep(2)

BASE_URL = "http://localhost:8001"

# Load a few negative and positive test cases
with open('reports/diabetes/negative/all_cases_full.json') as f:
    neg_cases_data = json.load(f)[:3]
    neg_cases = [case['input'] for case in neg_cases_data]

with open('reports/diabetes/positive/all_cases_full.json') as f:
    pos_cases_data = json.load(f)[:3]
    pos_cases = [case['input'] for case in pos_cases_data]

print("=== TESTING DIABETES MODEL ===\n")

# Test negative cases
print("Testing NEGATIVE cases (should predict 0):")
for i, case in enumerate(neg_cases):
    # Prepare the input from stored data
    try:
        response = requests.post(f"{BASE_URL}/predict/diabetes", json=case)
        if response.status_code == 200:
            result = response.json()
            print(f"  Case {i}: Stored Expected={neg_cases_data[i]['expected']}, API Prediction={result['prediction']}, Probability={result['probability']:.4f}")
        else:
            print(f"  Case {i}: API Error - {response.status_code}")
    except Exception as e:
        print(f"  Case {i}: Exception - {e}")

print("\nTesting POSITIVE cases (should predict 1):")
for i, case in enumerate(pos_cases):
    try:
        response = requests.post(f"{BASE_URL}/predict/diabetes", json=case)
        if response.status_code == 200:
            result = response.json()
            print(f"  Case {i}: Stored Expected={pos_cases_data[i]['expected']}, API Prediction={result['prediction']}, Probability={result['probability']:.4f}")
        else:
            print(f"  Case {i}: API Error - {response.status_code}")
    except Exception as e:
        print(f"  Case {i}: Exception - {e}")

# Test heart cases
print("\n\n=== TESTING HEART MODEL ===\n")

with open('reports/heart/negative/all_cases_full.json') as f:
    neg_cases_data_h = json.load(f)[:3]
    neg_cases_h = [case['input'] for case in neg_cases_data_h]

with open('reports/heart/positive/all_cases_full.json') as f:
    pos_cases_data_h = json.load(f)[:3]
    pos_cases_h = [case['input'] for case in pos_cases_data_h]

print("Testing NEGATIVE cases (should predict 0):")
for i, case in enumerate(neg_cases_h):
    try:
        response = requests.post(f"{BASE_URL}/predict/heart", json=case)
        if response.status_code == 200:
            result = response.json()
            print(f"  Case {i}: Stored Expected={neg_cases_data_h[i]['expected']}, API Prediction={result['prediction']}, Probability={result['probability']:.4f}")
        else:
            print(f"  Case {i}: API Error - {response.status_code}")
    except Exception as e:
        print(f"  Case {i}: Exception - {e}")

print("\nTesting POSITIVE cases (should predict 1):")
for i, case in enumerate(pos_cases_h):
    try:
        response = requests.post(f"{BASE_URL}/predict/heart", json=case)
        if response.status_code == 200:
            result = response.json()
            print(f"  Case {i}: Stored Expected={pos_cases_data_h[i]['expected']}, API Prediction={result['prediction']}, Probability={result['probability']:.4f}")
        else:
            print(f"  Case {i}: API Error - {response.status_code}")
    except Exception as e:
        print(f"  Case {i}: Exception - {e}")
