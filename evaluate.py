"""
evaluate.py — Production Evaluation Pipeline for AI JEE Agent
=============================================================

PURPOSE:
    This script sends 30 JEE-level queries to your FastAPI /chat endpoint,
    measures latency, stores responses, and leaves room for manual grading.

WHY WE MEASURE THESE METRICS:
    - Latency: In LLM systems, a slow response (>5s) means bad UX.
      We measure how fast your RAG pipeline retrieves + generates.
    - Response length: Very short responses may mean the LLM gave up or
      the retrieval found nothing relevant.
    - Correctness + Evidence: These are RAG-specific quality signals.
      If the LLM gives a correct answer BUT cites no evidence from your
      PDFs, it might be "hallucinating" — making things up from training
      data rather than your study material.

HOW THIS REFLECTS RAG PERFORMANCE:
    A good RAG system should:
      1. Retrieve the right chunks (evidence_present = Yes)
      2. Generate a correct answer based on those chunks (correctness = Correct)
      3. Do this fast (low latency)
    This script helps you measure all three dimensions.
"""

import json
import time
import csv
import os
import statistics
import requests

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG — Change BASE_URL if your server runs on a different host/port
# ──────────────────────────────────────────────────────────────────────────────
BASE_URL = "http://localhost:8000"
ENDPOINT = f"{BASE_URL}/query"          # NEW /query endpoint (see main.py update)
RESULTS_JSON = "evaluation_results.json"
RESULTS_CSV = "evaluation_results.csv"
REQUEST_TIMEOUT = 60                    # seconds before giving up on a request
EVAL_SESSION_ID = "eval_session"       # fixed session so history stays shared

# ──────────────────────────────────────────────────────────────────────────────
# 30 DIVERSE JEE QUERIES
# Covers: Physics, Chemistry, Math | Conceptual, Numerical, Theory
# ──────────────────────────────────────────────────────────────────────────────
JEE_QUERIES = [
    # ── PHYSICS: Conceptual ──────────────────────────────────────────────────
    "What is the principle of superposition of waves?",
    "Explain the difference between static and kinetic friction.",
    "What is the photoelectric effect and who explained it?",
    "Define electric flux and state Gauss's Law.",
    "What is the difference between EMF and terminal voltage of a cell?",

    # ── PHYSICS: Numerical ───────────────────────────────────────────────────
    "A body of mass 5 kg moves with velocity 10 m/s. What is its kinetic energy?",
    "A charge of 2 μC is placed 30 cm from another charge of 3 μC. Find the force between them.",
    "A wire carries 3 A of current through a 0.5 T magnetic field. Length of wire is 2 m. Find the force on the wire.",
    "An object is projected at 30° to the horizontal with speed 20 m/s. Find the range.",
    "What is the de Broglie wavelength of an electron moving at 1×10⁶ m/s?",

    # ── PHYSICS: Theory ──────────────────────────────────────────────────────
    "State and explain Newton's second law of motion.",
    "What are the conditions for total internal reflection?",
    "Explain the working principle of a transformer.",

    # ── CHEMISTRY: Conceptual ────────────────────────────────────────────────
    "What is Le Chatelier's principle? Give an example.",
    "Explain the difference between sigma and pi bonds.",
    "What is electronegativity and how does it vary across the periodic table?",
    "What is hybridization? Explain sp3 hybridization with an example.",
    "What are colligative properties? Name four of them.",

    # ── CHEMISTRY: Numerical ─────────────────────────────────────────────────
    "Calculate the molarity of a solution containing 20g of NaOH in 500mL of solution.",
    "What is the pH of a 0.01 M HCl solution?",
    "Find the empirical formula of a compound with 40% C, 6.7% H, and 53.3% O by mass.",

    # ── CHEMISTRY: Theory ────────────────────────────────────────────────────
    "Explain the mechanism of SN1 and SN2 reactions.",
    "What is the aufbau principle and how does it determine electron configuration?",
    "Explain Hund's rule of maximum multiplicity.",

    # ── MATH: Conceptual ─────────────────────────────────────────────────────
    "What is the geometric interpretation of a derivative?",
    "Explain the concept of limits in calculus.",
    "What is the difference between permutations and combinations?",

    # ── MATH: Numerical ──────────────────────────────────────────────────────
    "Find the roots of the quadratic equation: 2x² - 5x + 3 = 0.",
    "Differentiate f(x) = x³ sin(x) with respect to x.",
    "Find the integral of (3x² + 2x + 1) dx.",
]

assert len(JEE_QUERIES) == 30, "Ensure exactly 30 queries are defined!"


# ──────────────────────────────────────────────────────────────────────────────
# HEALTH CHECK — make sure server is live before starting
# ──────────────────────────────────────────────────────────────────────────────
def check_server_health():
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print(f"✅ Server is running at {BASE_URL}")
            return True
    except requests.exceptions.ConnectionError:
        pass
    print(f"❌ Could not connect to server at {BASE_URL}")
    print("   → Make sure your FastAPI server is running: uvicorn main:app --reload")
    return False


# ──────────────────────────────────────────────────────────────────────────────
# SINGLE QUERY FUNCTION
# ──────────────────────────────────────────────────────────────────────────────
def query_agent(question: str, session_id: str = EVAL_SESSION_ID) -> dict:
    """
    Sends one question to the /query endpoint.
    Returns a dict with: query, response, latency, retrieved_context, error.
    """
    payload = {"question": question, "session_id": session_id}

    start = time.time()
    try:
        resp = requests.post(ENDPOINT, json=payload, timeout=REQUEST_TIMEOUT)
        latency = round(time.time() - start, 3)

        if resp.status_code == 200:
            data = resp.json()
            return {
                "query": question,
                "response": data.get("answer", ""),
                "latency_seconds": latency,
                "response_length": len(data.get("answer", "")),
                "retrieved_context": data.get("retrieved_context", []),
                # ── Manual evaluation fields (fill these in after running) ──
                "correctness": "",          # Correct / Partially Correct / Incorrect
                "evidence_present": "",     # Yes / No
                "notes": "",               # Optional: your personal notes
                "error": None,
                "http_status": resp.status_code,
            }
        else:
            return {
                "query": question,
                "response": "",
                "latency_seconds": latency,
                "response_length": 0,
                "retrieved_context": [],
                "correctness": "",
                "evidence_present": "",
                "notes": "",
                "error": f"HTTP {resp.status_code}: {resp.text}",
                "http_status": resp.status_code,
            }

    except requests.exceptions.Timeout:
        latency = round(time.time() - start, 3)
        return {
            "query": question,
            "response": "",
            "latency_seconds": latency,
            "response_length": 0,
            "retrieved_context": [],
            "correctness": "",
            "evidence_present": "",
            "notes": "",
            "error": "Request timed out",
            "http_status": None,
        }
    except Exception as e:
        latency = round(time.time() - start, 3)
        return {
            "query": question,
            "response": "",
            "latency_seconds": 0,
            "response_length": 0,
            "retrieved_context": [],
            "correctness": "",
            "evidence_present": "",
            "notes": "",
            "error": str(e),
            "http_status": None,
        }


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY STATISTICS
# ──────────────────────────────────────────────────────────────────────────────
def compute_summary(results: list) -> dict:
    """
    Computes aggregate statistics from the results list.
    NOTE: correctness/evidence columns will be empty until manual labeling.
    """
    latencies = [r["latency_seconds"] for r in results if r["error"] is None]
    errors = [r for r in results if r["error"] is not None]

    # Manual grading stats (computed after you fill in the fields)
    correct_count = sum(1 for r in results if r["correctness"] == "Correct")
    partial_count = sum(1 for r in results if r["correctness"] == "Partially Correct")
    incorrect_count = sum(1 for r in results if r["correctness"] == "Incorrect")
    evidence_yes = sum(1 for r in results if r["evidence_present"] == "Yes")
    labeled = sum(1 for r in results if r["correctness"] != "")

    summary = {
        "total_queries": len(results),
        "successful_responses": len(latencies),
        "failed_responses": len(errors),
        "latency": {
            "average_seconds": round(statistics.mean(latencies), 3) if latencies else 0,
            "min_seconds": round(min(latencies), 3) if latencies else 0,
            "max_seconds": round(max(latencies), 3) if latencies else 0,
            "median_seconds": round(statistics.median(latencies), 3) if latencies else 0,
            "std_dev_seconds": round(statistics.stdev(latencies), 3) if len(latencies) > 1 else 0,
        },
        "response_length": {
            "average_chars": round(
                statistics.mean([r["response_length"] for r in results if r["response_length"] > 0]), 1
            ) if any(r["response_length"] > 0 for r in results) else 0,
        },
        "manual_evaluation": {
            "labeled_count": labeled,
            "correct": correct_count,
            "partially_correct": partial_count,
            "incorrect": incorrect_count,
            "accuracy_percent": round(correct_count / labeled * 100, 1) if labeled > 0 else "Run manually first",
            "evidence_present_count": evidence_yes,
            "evidence_percent": round(evidence_yes / labeled * 100, 1) if labeled > 0 else "Run manually first",
        },
    }
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# SAVE TO CSV
# ──────────────────────────────────────────────────────────────────────────────
def save_csv(results: list, filepath: str):
    """Saves results as a CSV — easier to open in Excel or Google Sheets for manual grading."""
    if not results:
        return

    fieldnames = [
        "query", "response", "latency_seconds", "response_length",
        "retrieved_context", "correctness", "evidence_present", "notes", "error", "http_status"
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Flatten retrieved_context list to a string for CSV
            row_copy = row.copy()
            if isinstance(row_copy.get("retrieved_context"), list):
                row_copy["retrieved_context"] = " || ".join(row_copy["retrieved_context"])
            writer.writerow(row_copy)

    print(f"📄 CSV saved → {filepath}")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ──────────────────────────────────────────────────────────────────────────────
def run_evaluation():
    print("\n" + "="*65)
    print("   🔬 AI JEE AGENT — EVALUATION PIPELINE")
    print("="*65)

    if not check_server_health():
        return

    print(f"\n📋 Running {len(JEE_QUERIES)} JEE queries...\n")

    results = []
    for i, query in enumerate(JEE_QUERIES, 1):
        print(f"[{i:02d}/{len(JEE_QUERIES)}] {query[:70]}...")
        result = query_agent(query)

        if result["error"]:
            print(f"       ❌ Error: {result['error']}")
        else:
            print(f"       ✅ Latency: {result['latency_seconds']}s | Length: {result['response_length']} chars")

        results.append(result)

        # Small delay between requests to avoid rate-limiting
        if i < len(JEE_QUERIES):
            time.sleep(0.5)

    # ── Save results ──────────────────────────────────────────────────────────
    summary = compute_summary(results)

    output = {
        "summary": summary,
        "results": results,
    }

    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n💾 JSON saved → {RESULTS_JSON}")

    save_csv(results, RESULTS_CSV)

    # ── Print summary to terminal ─────────────────────────────────────────────
    print("\n" + "="*65)
    print("   📊 EVALUATION SUMMARY")
    print("="*65)
    print(f"  Total queries      : {summary['total_queries']}")
    print(f"  Successful         : {summary['successful_responses']}")
    print(f"  Failed             : {summary['failed_responses']}")
    print(f"  Avg latency        : {summary['latency']['average_seconds']}s")
    print(f"  Min latency        : {summary['latency']['min_seconds']}s")
    print(f"  Max latency        : {summary['latency']['max_seconds']}s")
    print(f"  Avg response length: {summary['response_length']['average_chars']} chars")
    print()
    print("  📝 MANUAL GRADING INSTRUCTIONS:")
    print(f"     Open '{RESULTS_CSV}' in Excel or Google Sheets.")
    print("     For each row, fill in:")
    print("       • correctness    → 'Correct' / 'Partially Correct' / 'Incorrect'")
    print("       • evidence_present → 'Yes' / 'No' (did the answer cite study material?)")
    print("     Then re-run: python evaluate.py --summary-only")
    print("="*65 + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# SUMMARY-ONLY MODE — run after manual grading
# ──────────────────────────────────────────────────────────────────────────────
def run_summary_only():
    """Reads the existing JSON and re-computes summary after manual labeling."""
    if not os.path.exists(RESULTS_JSON):
        print(f"❌ {RESULTS_JSON} not found. Run the full evaluation first.")
        return

    with open(RESULTS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = data.get("results", [])
    summary = compute_summary(results)

    print("\n" + "="*65)
    print("   📊 UPDATED SUMMARY (after manual grading)")
    print("="*65)
    ml = summary["manual_evaluation"]
    print(f"  Labeled            : {ml['labeled_count']} / {summary['total_queries']}")
    print(f"  Correct            : {ml['correct']}")
    print(f"  Partially Correct  : {ml['partially_correct']}")
    print(f"  Incorrect          : {ml['incorrect']}")
    print(f"  Accuracy           : {ml['accuracy_percent']}%")
    print(f"  Evidence Present   : {ml['evidence_present_count']} ({ml['evidence_percent']}%)")
    print("="*65 + "\n")

    # Overwrite JSON with updated summary
    data["summary"] = summary
    with open(RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"✅ Updated summary saved to {RESULTS_JSON}")


# ──────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    if "--summary-only" in sys.argv:
        run_summary_only()
    else:
        run_evaluation()
