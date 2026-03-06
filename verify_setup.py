#!/usr/bin/env python3
"""
Verification script to check if everything is set up correctly.
"""

import os
import sys
from pathlib import Path

def check_python():
    """Check Python version."""
    print("\n📍 Python Check:")
    version = sys.version_info
    print(f"  Version: {version.major}.{version.minor}.{version.micro}")
    if version.major >= 3 and version.minor >= 8:
        print(f"  ✅ OK")
        return True
    else:
        print(f"  ❌ Need Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n📦 Dependencies Check:")
    required = [
        'langgraph',
        'langchain',
        'langchain_community',
        'qdrant_client',
        'groq',
        'fastembed',
    ]
    
    all_ok = True
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
            print(f"  ✅ {pkg}")
        except ImportError:
            print(f"  ❌ {pkg} - Run: pip install -r requirements.txt")
            all_ok = False
    
    return all_ok

def check_directories():
    """Check if required directories exist."""
    print("\n📁 Directories Check:")
    dirs = ['graph', 'prompts', 'tools', 'utils', 'output', 'qdrant_db']
    
    for d in dirs:
        if Path(d).exists():
            print(f"  ✅ {d}/")
        else:
            os.makedirs(d, exist_ok=True)
            print(f"  ⚠️  {d}/ - Created")
    
    return True

def check_files():
    """Check if required files exist."""
    print("\n📄 Files Check:")
    files = [
        'config.py',
        'main.py',
        'app.py',
        'state.py',
        'memory.py',
        '.env',
        'requirements.txt'
    ]
    
    all_ok = True
    for f in files:
        if Path(f).exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - Missing!")
            all_ok = False
    
    return all_ok

def check_env():
    """Check .env file."""
    print("\n🔑 Environment Check:")
    if not Path('.env').exists():
        print("  ❌ .env file not found")
        print("     Create it: cp .env.example .env")
        return False
    
    # Load .env
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check key variables
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        print(f"  ✅ GROQ_API_KEY: Set")
    else:
        print(f"  ⚠️  GROQ_API_KEY: Not set")
    
    tavily_key = os.getenv('TAVILY_API_KEY')
    if tavily_key:
        print(f"  ✅ TAVILY_API_KEY: Set")
    else:
        print(f"  ⚠️  TAVILY_API_KEY: Not set (optional)")
    
    langchain_trace = os.getenv('LANGCHAIN_TRACING_V2', 'false')
    if langchain_trace.lower() == 'false':
        print(f"  ✅ LANGCHAIN_TRACING_V2: Disabled")
    else:
        print(f"  ⚠️  LANGCHAIN_TRACING_V2: Enabled (may cause auth errors)")
    
    return True

def check_config():
    """Check if config can be imported."""
    print("\n⚙️  Configuration Check:")
    try:
        from config import Config
        print(f"  ✅ Model: {Config.MODEL_NAME}")
        print(f"  ✅ GROQ Model: {Config.GROQ_MODEL_NAME}")
        print(f"  ✅ GROQ API URL: {Config.GROQ_API_URL}")
        return True
    except Exception as e:
        print(f"  ❌ Config Error: {e}")
        return False

def check_groq():
    """Check if Groq API key is set."""
    print("\n🤖 Groq Check:")
    key = os.getenv('GROQ_API_KEY')
    if key and key.strip():
        print(f"  ✅ GROQ_API_KEY: Set")
        return True
    else:
        print(f"  ❌ GROQ_API_KEY: Not set")
        return False

def check_memory():
    """Check if memory/Qdrant is working."""
    print("\n💾 Memory Check:")
    try:
        from memory import memory
        print(f"  ✅ Memory manager initialized")
        # Try to get context
        result = memory.get_context("test query")
        print(f"  ✅ Qdrant collection accessible")
        return True
    except Exception as e:
        print(f"  ⚠️  Memory Error: {e}")
        print(f"     Try: rm -rf qdrant_db && mkdir qdrant_db")
        return False

def main():
    """Run all checks."""
    print("\n" + "="*60)
    print("Developer Research AI Agent - Setup Verification")
    print("="*60)
    
    checks = [
        ("Python", check_python),
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Files", check_files),
        ("Environment", check_env),
        ("Configuration", check_config),
        ("Memory", check_memory),
    ]
    
    results = {}
    for name, check_func in checks:
        try:
            results[name] = check_func()
        except Exception as e:
            print(f"  ❌ Error during {name} check: {e}")
            results[name] = False
    
    # Summary
    print("\n" + "="*60)
    print("✅ Summary:")
    print("="*60)
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    
    for name, result in results.items():
        status = "✅" if result else "⚠️ "
        print(f"{status} {name}")
    
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n🎉 All checks passed! You're ready to run:")
        print("   python3 main.py")
        return 0
    elif passed >= total - 2:
        print("\n⚠️  Some optional checks failed, but you might still run the agent.")
        print("   Make sure Qdrant is accessible and no other instances are running")
        return 1
    else:
        print("\n❌ Please fix the errors above and try again.")
        print("   See ERROR_FIX.md for detailed troubleshooting.")
        return 2

if __name__ == "__main__":
    sys.exit(main())
