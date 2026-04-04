import json
import requests
import time

time.sleep(1)
BASE_URL = "http://localhost:8001"

with open('reports/diabetes/negative/all_cases_full.json') as f:
    neg_cases_data = json.load(f)[:5]
    neg_cases = [case['input'] for case in neg_cases_data]

with open('reports/diabetes/positive/all_cases_full.json') as f:
    pos_cases_data = json.load(f)[:5]
    pos_cases = [case['input'] for case in pos_cases_data]

with open('reports/heart/negative/all_cases_full.json') as f:
    neg_cases_data_h = json.load(f)[:5]
    neg_cases_h = [case['input'] for case in neg_cases_data_h]

with open('reports/heart/positive/all_cases_full.json') as f:
    pos_cases_data_h = json.load(f)[:5]
    pos_cases_h = [case['input'] for case in pos_cases_data_h]

print("\n" + "="*70)
print("HEART MODEL TEST (Threshold 0.20)")
print("="*70 + "\n")

print("NEGATIVE cases (should predict 0/Negative):")
for i, case in enumerate(neg_cases_h):
    response = requests.post(f"{BASE_URL}/predict/heart", json=case)
    if response.status_code == 200:
        result = response.json()
        expected = "Negative" if neg_cases_data_h[i]['expected'] == 0 else "Positive"
        status = "✓" if (expected == "Negative" and result['prediction'] == "Negative") or (expected == "Positive" and result['prediction'] == "Positive") else "✗"
        print(f"  {status} Case {i}: Expected={expected}, Predicted={result['prediction']}, Prob={result['probability']:.4f}")

print("\nPOSITIVE cases (should predict 1/Positive):")
for i, case in enumerate(pos_cases_h):
    response = requests.post(f"{BASE_URL}/predict/heart", json=case)
    if response.status_code == 200:
        result = response.json()
        expected = "Negative" if pos_cases_data_h[i]['expected'] == 0 else "Positive"
        status = "✓" if (expected == "Negative" and result['prediction'] == "Negative") or (expected == "Positive" and result['prediction'] == "Positive") else "✗"
        print(f"  {status} Case {i}: Expected={expected}, Predicted={result['prediction']}, Prob={result['probability']:.4f}")

print("\n" + "="*70)
print("DIABETES MODEL TEST (Threshold 0.40)")
print("="*70 + "\n")

print("NEGATIVE cases (should predict 0/Negative):")
for i, case in enumerate(neg_cases):
    response = requests.post(f"{BASE_URL}/predict/diabetes", json=case)
    if response.status_code == 200:
        result = response.json()
        expected = "Negative" if neg_cases_data[i]['expected'] == 0 else "Positive"
        status = "✓" if (expected == "Negative" and result['prediction'] == "Negative") or (expected == "Positive" and result['prediction'] == "Positive") else "✗"
        print(f"  {status} Case {i}: Expected={expected}, Predicted={result['prediction']}, Prob={result['probability']:.4f}")

print("\nPOSITIVE cases (should predict 1/Positive):")
for i, case in enumerate(pos_cases):
    response = requests.post(f"{BASE_URL}/predict/diabetes", json=case)
    if response.status_code == 200:
        result = response.json()
        expected = "Negative" if pos_cases_data[i]['expected'] == 0 else "Positive"
        status = "✓" if (expected == "Negative" and result['prediction'] == "Negative") or (expected == "Positive" and result['prediction'] == "Positive") else "✗"
        print(f"  {status} Case {i}: Expected={expected}, Predicted={result['prediction']}, Prob={result['probability']:.4f}")
