#!/usr/bin/env python3
"""
Dependency checker for Developer Research AI Agent
Run this script to verify all dependencies are properly installed.
"""

import sys
import importlib
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_module(module_name, package_name=None):
    """Check if a module is installed."""
    try:
        importlib.import_module(module_name)
        print(f"{GREEN}✓{RESET} {module_name}")
        return True
    except ImportError:
        pkg = package_name or module_name
        print(f"{RED}✗{RESET} {module_name} - Install with: pip install {pkg}")
        return False

def check_file(file_path):
    """Check if a file exists."""
    if Path(file_path).exists():
        print(f"{GREEN}✓{RESET} {file_path}")
        return True
    else:
        print(f"{RED}✗{RESET} {file_path} - Not found")
        return False

def main():
    print(f"\n{BLUE}Developer Research AI Agent - Dependency Check{RESET}")
    print("=" * 50)
    
    # Check Python version
    print(f"\n{BLUE}Python Environment:{RESET}")
    python_version = sys.version_info
    if python_version.major >= 3 and python_version.minor >= 8:
        print(f"{GREEN}✓{RESET} Python {python_version.major}.{python_version.minor}.{python_version.micro}")
    else:
        print(f"{RED}✗{RESET} Python 3.8+ required (you have {python_version.major}.{python_version.minor})")
        return False
    
    # Check core dependencies
    print(f"\n{BLUE}Core Dependencies:{RESET}")
    core_deps = [
        ("langchain", "langchain"),
        ("langgraph", "langgraph"),
        ("qdrant_client", "qdrant-client"),
        ("dotenv", "python-dotenv"),
        ("groq", "groq"),
    ]
    
    core_ok = all(check_module(mod, pkg) for mod, pkg in core_deps)
    
    # Check optional dependencies
    print(f"\n{BLUE}Optional Dependencies:{RESET}")
    optional_deps = [
        ("streamlit", "streamlit"),
        ("tavily", "tavily-python"),
        ("duckduckgo_search", "duckduckgo-search"),
    ]
    
    optional_ok = True
    for mod, pkg in optional_deps:
        try:
            importlib.import_module(mod)
            print(f"{GREEN}✓{RESET} {mod}")
        except ImportError:
            print(f"{YELLOW}⚠{RESET} {mod} (optional) - Install with: pip install {pkg}")
            optional_ok = False
    
    # Check project structure
    print(f"\n{BLUE}Project Structure:{RESET}")
    required_files = [
        "main.py",
        "app.py",
        "config.py",
        "state.py",
        "memory.py",
        "requirements.txt",
        "graph/nodes_pre.py",
        "graph/nodes_exec.py",
        "graph/nodes_post.py",
        "prompts/research_prompts.py",
        "tools/search_tools.py",
    ]
    
    files_ok = all(check_file(f) for f in required_files)
    
    # Check directories
    print(f"\n{BLUE}Required Directories:{RESET}")
    import os
    dirs = ["graph", "prompts", "tools", "utils", "output", "qdrant_db"]
    for d in dirs:
        if Path(d).exists():
            print(f"{GREEN}✓{RESET} {d}/")
        else:
            os.makedirs(d, exist_ok=True)
            print(f"{YELLOW}⚠{RESET} {d}/ - Created")
    
    # Check .env file
    print(f"\n{BLUE}Configuration:{RESET}")
    dot_env_ok = check_file(".env")
    if not dot_env_ok:
        print(f"{YELLOW}ℹ{RESET} .env not found. Copy from .env.example:")
        print(f"  {YELLOW}cp .env.example .env{RESET}")
    
    # Summary
    print(f"\n{BLUE}Summary:{RESET}")
    print("-" * 50)
    
    if core_ok and files_ok:
        print(f"{GREEN}✓ All core requirements met!{RESET}")
        if optional_ok:
            print(f"{GREEN}✓ All optional dependencies installed!{RESET}")
        else:
            print(f"{YELLOW}⚠ Some optional dependencies missing (won't affect core functionality){RESET}")
        
        print(f"\n{GREEN}✓ Ready to run!{RESET}")
        print(f"\nStart with: {BLUE}python3 main.py{RESET}")
        return True
    else:
        print(f"{RED}✗ Missing required dependencies or files{RESET}")
        print(f"\nFix by running: {YELLOW}pip install -r requirements.txt{RESET}")
        return False

if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    success = main()
    sys.exit(0 if success else 1)
