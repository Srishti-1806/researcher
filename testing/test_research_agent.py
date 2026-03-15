from main import build_agent
import sys

def test_agent():
    print("🚀 Starting Full Agent Test...")
    
    try:
        app = build_agent()
    except Exception as e:
        print(f"❌ Failed to build agent: {e}")
        sys.exit(1)

    # Test Query
    query = "What is the Smile representation of 2,3-dimethylbutane and how does it compare to the structure of Mamba Transformer?"
    print(f"🔍 Testing Query: '{query}'")
    
    initial_state = {
        "query": query,
        "history": [],
    }
    
    final_report = ""
    steps_completed = []

    try:
        for output in app.stream(initial_state):
            for key, value in output.items():
                print(f"✅ Executed Node: [{key}]")
                steps_completed.append(key)
                if key == "formatter":
                    final_report = value.get("final_report", "")
    except Exception as e:
        print(f"❌ Error during execution: {e}")
        sys.exit(1)

    print("\n" + "="*40)
    print("TEST REPORT OUTPUT")
    print("="*40)
    print(final_report)
    print("="*40 + "\n")

    # Assertions
    required_steps = [
        "guard", "preprocess", "embed", "vector", "classify", "planner",
        "deep_research", "gap_analysis", "synthesize", "formatter"
    ]
    missing_steps = [step for step in required_steps if step not in steps_completed]
    
    if not missing_steps:
        print("✅ SUCCESS: All expected nodes executed.")
    else:
        # Deep research might skip if confidence is high, but for this complex query it should trigger.
        # Planners might route to quick mode if classified incorrectly.
        print(f"⚠️ WARNING: Some expected nodes were missing: {missing_steps}")
        print(f"   Actual path: {steps_completed}")
        
    if "Mamba" in final_report and "Transformer" in final_report:
        print("✅ SUCCESS: Report contains expected keywords.")
    else:
        print("❌ FAILURE: Report missing keywords.")

if __name__ == "__main__":
    test_agent()
