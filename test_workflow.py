#!/usr/bin/env python3
"""
Test the chemical translation workflow programmatically
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from state import AgentState
from graph.nodes_exec import deep_mode_orchestrator

def test_translation(input_value: str, input_format: str, output_format: str):
    """Test a single translation."""
    print(f"\n🧪 Testing: {input_value} ({input_format} → {output_format})")

    # Create initial state
    state = AgentState(
        query=f"Translate {input_value} from {input_format} to {output_format}",
        input_value=input_value,
        input_format=input_format,
        output_format=output_format,
        iterations=0,
        research_data=[],
        history=[]
    )

    # Run the orchestrator
    result = deep_mode_orchestrator(state)

    print(f"📝 Final Translation: {result.get('final_translation', 'N/A')}")
    print(f"🔄 Reverse Translation: {result.get('reverse_translation', 'N/A')}")
    print(".3f")
    print(".3f")
    print(f"🔄 Iterations: {result.get('iterations', 0)}")
    print(f"⚠️  Gaps: {result.get('gaps', [])}")

    # Add test metadata to result
    result['input_value'] = input_value
    result['input_format'] = input_format
    result['output_format'] = output_format

    return result

def main():
    """Test various chemical translations."""
    print("🚀 Testing Enhanced Chemical Translation System")
    print("=" * 50)

    test_cases = [
        ("CC(C)C", "SMILES", "IUPAC"),  # 2-methylpropane
        ("CCO", "SMILES", "IUPAC"),     # ethanol
        ("ethane", "IUPAC", "SMILES"),  # ethane
        ("2-methylpropane", "IUPAC", "SMILES"),  # back to SMILES
    ]

    results = []

    for input_val, in_fmt, out_fmt in test_cases:
        try:
            result = test_translation(input_val, in_fmt, out_fmt)
            results.append(result)
        except Exception as e:
            print(f"❌ Error testing {input_val}: {e}")
            results.append({"error": str(e), "input": input_val})

    # Summary
    print("\n📊 Summary")
    print("-" * 30)

    successful = sum(1 for r in results if r.get('confidence_score', 0) > 0.8)
    total = len(results)

    print(f"Total Tests: {total}")
    print(f"High Confidence (>0.8): {successful}")
    print(f"Success Rate: {successful/total*100:.1f}%" if total > 0 else "N/A")

    # Check the problematic case from the original issue
    print("\n🎯 Original Issue Check")
    print("-" * 25)
    problematic_result = None
    for r in results:
        if isinstance(r, dict) and r.get('input_value') == 'CC(C)C' and r.get('input_format') == 'SMILES':
            problematic_result = r
            break

    if problematic_result:
        translation = problematic_result.get('final_translation', '')
        if '2-methylpropane' in translation.lower():
            print("✅ FIXED: CC(C)C now correctly translates to '2-methylpropane'")
            print(f"   Previous issue: was giving 'C(C)(C)c' or similar incorrect SMILES")
            print(f"   Now gives: '{translation}'")
        else:
            print(f"❌ STILL BROKEN: CC(C)C translates to '{translation}' instead of '2-methylpropane'")
    else:
        print("⚠️ Could not find test for the original issue")

if __name__ == "__main__":
    main()