import re
import json
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent
TEST_FILE = ROOT / "TEST_CASES.txt"
OUT_FILE = ROOT / "test_results_tabular.json"


def _parse_model_section(model: str, section_text: str):
    cases = []
    lines = section_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        m = re.match(r"^TEST CASE\s+(\d+):\s*(.+)$", line.strip())
        if not m:
            i += 1
            continue

        tc_num = int(m.group(1))
        title = m.group(2).strip()
        fields = {}
        expected = ""

        j = i + 1
        while j < len(lines) and "Fields:" not in lines[j]:
            j += 1
        j += 1

        while j < len(lines) and "Expected Output:" not in lines[j]:
            fl = lines[j]
            fm = re.match(r"^\s{2}([A-Za-z_][A-Za-z0-9_]*):\s*(.+)$", fl)
            if fm:
                key = fm.group(1)
                raw = fm.group(2).split("#")[0].strip().rstrip(",")
                if raw.startswith('"') and raw.endswith('"'):
                    value = raw[1:-1]
                elif raw.lower() in {"true", "false"}:
                    value = raw.lower() == "true"
                else:
                    try:
                        if re.fullmatch(r"-?\d+", raw):
                            value = int(raw)
                        else:
                            value = float(raw)
                    except Exception:
                        value = raw
                fields[key] = value
            j += 1

        if j < len(lines) and "Expected Output:" in lines[j]:
            expected = lines[j].split("Expected Output:", 1)[1].strip()

        cases.append(
            {
                "model": model,
                "tc": tc_num,
                "title": title,
                "fields": fields,
                "expected_text": expected,
            }
        )
        i = j + 1

    return cases


def parse_cases(text: str):
    diabetes_split = text.split("DIABETES MODEL TEST CASES", 1)
    heart_split = text.split("HEART DISEASE MODEL TEST CASES", 1)

    if len(diabetes_split) < 2 or len(heart_split) < 2:
        return []

    diabetes_block = diabetes_split[1].split("HEART DISEASE MODEL TEST CASES", 1)[0]
    heart_block = heart_split[1].split("EDGE CASE CATEGORIES TESTED", 1)[0]

    cases = []
    cases.extend(_parse_model_section("diabetes", diabetes_block))
    cases.extend(_parse_model_section("heart", heart_block))
    return cases


def expected_kind(expected_text: str):
    up = expected_text.upper()
    has_pos = "POSITIVE" in up
    has_neg = "NEGATIVE" in up
    if has_pos and not has_neg:
        return "Positive"
    if has_neg and not has_pos:
        return "Negative"
    return "Mixed"


def main():
    text = TEST_FILE.read_text(encoding="utf-8", errors="ignore")
    cases = parse_cases(text)
    results = []

    for case in cases:
        url = f"http://127.0.0.1:8001/predict/{case['model']}"
        rec = {
            "model": case["model"],
            "tc": case["tc"],
            "title": case["title"],
            "expected": expected_kind(case["expected_text"]),
            "expected_text": case["expected_text"],
        }
        try:
            resp = requests.post(url, json=case["fields"], timeout=45)
            rec["status_code"] = resp.status_code
            if resp.status_code == 200:
                body = resp.json()
                rec["actual"] = body.get("prediction")
                rec["probability"] = body.get("probability")
                rec["pass"] = rec["expected"] == "Mixed" or rec["actual"] == rec["expected"]
            else:
                rec["actual"] = f"HTTP {resp.status_code}"
                rec["probability"] = None
                rec["pass"] = False
        except Exception as exc:
            rec["status_code"] = None
            rec["actual"] = f"ERROR: {type(exc).__name__}"
            rec["probability"] = None
            rec["pass"] = False
        results.append(rec)

    OUT_FILE.write_text(json.dumps(results, indent=2), encoding="utf-8")

    for model in ["diabetes", "heart"]:
        subset = [r for r in results if r["model"] == model]
        passed = sum(1 for r in subset if r["pass"])
        print(f"[{model.upper()}] {passed}/{len(subset)} matched expectation")
        for r in subset:
            prob = "-" if r["probability"] is None else f"{float(r['probability']):.4f}"
            status = "PASS" if r["pass"] else "FAIL"
            print(
                f"TC{r['tc']:02d} | expected={r['expected']:<8} actual={str(r['actual']):<8} prob={prob} | {status} | {r['title']}"
            )

    print(f"\nSaved detailed JSON: {OUT_FILE}")


if __name__ == "__main__":
    main()
