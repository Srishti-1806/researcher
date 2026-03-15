#!/usr/bin/env python3
"""
Chemical Translation Testing & Benchmarking Script
Tests the enhanced chemical translation system with various molecules.
"""

import sys
import os
import time
import json
from typing import Dict, List, Tuple

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.nodes_exec import (
    canonicalize_smiles,
    compute_similarity,
    _is_valid_smiles,
    validate_iupac_name
)
from utils.groq_client import get_llm

def test_canonicalization():
    """Test SMILES canonicalization."""
    print("🧪 Testing SMILES Canonicalization")
    test_cases = [
        ("CC(C)C", "CC(C)C"),  # 2-methylpropane (already canonical)
        (" C(C)(C)C ", "CC(C)C"),  # Alternative form with whitespace
        ("CCO", "CCO"),  # Ethanol
        ("OCC", "CCO"),  # Ethanol (different order)
    ]

    results = []
    for input_smiles, expected in test_cases:
        canonical = canonicalize_smiles(input_smiles)
        success = canonical == expected if canonical else False
        results.append({
            "input": input_smiles,
            "expected": expected,
            "output": canonical,
            "success": success
        })
        print(f"  {input_smiles} → {canonical} {'✅' if success else '❌'}")

    return results

def test_similarity():
    """Test molecular similarity computation."""
    print("\n🧪 Testing Molecular Similarity")
    test_cases = [
        ("CC(C)C", "CC(C)C", 1.0),  # Identical
        ("CCO", "CCO", 1.0),  # Identical
        ("CC(C)C", "CCC", 0.5),  # Similar but different
    ]

    results = []
    for smiles1, smiles2, expected_range in test_cases:
        similarity = compute_similarity(smiles1, smiles2)
        success = expected_range - 0.1 <= similarity <= expected_range + 0.1
        results.append({
            "smiles1": smiles1,
            "smiles2": smiles2,
            "similarity": similarity,
            "expected_range": expected_range,
            "success": success
        })
        print(f"  {smiles1} vs {smiles2}: {similarity:.3f} {'✅' if success else '❌'}")

    return results

def test_validation():
    """Test SMILES and IUPAC validation."""
    print("\n🧪 Testing Validation")
    smiles_cases = [
        ("CC(C)C", True),  # Valid
        ("invalid", False),  # Invalid
        ("CCO", True),  # Valid
        ("C(C)(C)c", False),  # Invalid (aromatic carbon issue)
    ]

    iupac_cases = [
        ("2-methylpropane", True),
        ("ethane", True),
        ("", False),
        ("abc", False),
    ]

    results = []

    print("  SMILES Validation:")
    for smiles, expected in smiles_cases:
        valid = _is_valid_smiles(smiles)
        success = valid == expected
        results.append({
            "type": "SMILES",
            "input": smiles,
            "valid": valid,
            "expected": expected,
            "success": success
        })
        print(f"    {smiles}: {valid} {'✅' if success else '❌'}")

    print("  IUPAC Validation:")
    for name, expected in iupac_cases:
        valid = validate_iupac_name(name)
        success = valid == expected
        results.append({
            "type": "IUPAC",
            "input": name,
            "valid": valid,
            "expected": expected,
            "success": success
        })
        print(f"    {name}: {valid} {'✅' if success else '❌'}")

    return results

def benchmark_translation():
    """Benchmark the translation system with sample molecules."""
    print("\n📊 Benchmarking Translation System")

    # Initialize LLM
    llm = get_llm()
    if not llm:
        print("❌ LLM not available for benchmarking")
        return []

    test_cases = [
        {"input": "CC(C)C", "input_format": "SMILES", "expected_output": "2-methylpropane", "output_format": "IUPAC"},
        {"input": "CCO", "input_format": "SMILES", "expected_output": "ethanol", "output_format": "IUPAC"},
        {"input": "ethane", "input_format": "IUPAC", "expected_output": "CC", "output_format": "SMILES"},
    ]

    results = []

    for i, case in enumerate(test_cases):
        print(f"  Test {i+1}: {case['input']} ({case['input_format']} → {case['output_format']})")

        # Canonicalize input
        input_value = case['input']
        if case['input_format'].upper() == "SMILES":
            canonical = canonicalize_smiles(input_value)
            if canonical:
                input_value = canonical

        # Generate translation
        start_time = time.time()

        prompt = f"Translate {case['input_format']} '{input_value}' to {case['output_format']}. Output only the result:"
        try:
            output = llm.generate(prompt, temperature=0.1, max_new_tokens=50)
            output = str(output).strip()
            inference_time = time.time() - start_time

            # Validate output
            if case['output_format'].upper() == "SMILES":
                valid = _is_valid_smiles(output)
            else:
                valid = validate_iupac_name(output)

            # Compute similarity/accuracy
            if case['output_format'].upper() == "SMILES" and case['input_format'].upper() == "SMILES":
                similarity = compute_similarity(input_value, output)
            else:
                similarity = 1.0 if output.lower() == case['expected_output'].lower() else 0.0

            success = valid and similarity > 0.8

            results.append({
                "test_case": case,
                "output": output,
                "inference_time": inference_time,
                "valid": valid,
                "similarity": similarity,
                "success": success
            })

            print(f"    Output: {output}")
            print(".3f")
            print(f"    Result: {'✅ PASS' if success else '❌ FAIL'}")

        except Exception as e:
            print(f"    Error: {e}")
            results.append({
                "test_case": case,
                "error": str(e),
                "success": False
            })

    return results

def main():
    """Run all tests and benchmarks."""
    print("🚀 Chemical Translation System - Testing & Benchmarking")
    print("=" * 60)

    all_results = {}

    # Run tests
    all_results['canonicalization'] = test_canonicalization()
    all_results['similarity'] = test_similarity()
    all_results['validation'] = test_validation()
    all_results['benchmark'] = benchmark_translation()

    # Summary
    print("\n📈 Summary")
    print("-" * 30)

    total_tests = 0
    passed_tests = 0

    for category, results in all_results.items():
        if category == 'benchmark':
            for result in results:
                total_tests += 1
                if result.get('success', False):
                    passed_tests += 1
        else:
            for result in results:
                total_tests += 1
                if result.get('success', False):
                    passed_tests += 1

    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(".1f")

    # Save results
    with open('testing/benchmark_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print("\n💾 Results saved to testing/benchmark_results.json")

if __name__ == "__main__":
    main()