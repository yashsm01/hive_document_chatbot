#!/usr/bin/env python3
"""
Code Search Tool - Search your codebase using MCP Server

Ask questions like:
- "Where is the function get_llm_provider?"
- "Find all files using LiteLLMProvider"
- "Show me the GraphExecutor class"
"""

import os
import sys
import json
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'tools' / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'core'))

# Load .env
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key] = value.strip('"').strip("'")


def search_code(query: str, repo_path: str = None, extensions: list = None) -> list:
    """
    Search for code patterns in the repository.
    
    Args:
        query: Search term (function name, class name, etc.)
        repo_path: Path to repository (default: current hive repo)
        extensions: File extensions to search (default: .py)
    
    Returns:
        List of matches with file, line number, and content
    """
    if repo_path is None:
        repo_path = Path(__file__).parent.parent.parent
    
    if extensions is None:
        extensions = ['.py']
    
    results = []
    repo_path = Path(repo_path)
    
    for ext in extensions:
        for file_path in repo_path.rglob(f'*{ext}'):
            # Skip hidden and venv directories
            if any(part.startswith('.') or part in ('venv', '__pycache__', 'node_modules') 
                   for part in file_path.parts):
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    for line_num, line in enumerate(f, 1):
                        if query.lower() in line.lower():
                            results.append({
                                'file': str(file_path.relative_to(repo_path)),
                                'line': line_num,
                                'content': line.strip()[:200]
                            })
            except Exception:
                continue
    
    return results[:50]  # Limit results


def find_function(name: str, repo_path: str = None) -> list:
    """Find a function definition by name."""
    results = []
    repo_path = Path(repo_path) if repo_path else Path(__file__).parent.parent.parent
    
    for file_path in repo_path.rglob('*.py'):
        if any(part.startswith('.') or part in ('venv', '__pycache__') 
               for part in file_path.parts):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if f'def {name}' in line or f'async def {name}' in line:
                        results.append({
                            'file': str(file_path.relative_to(repo_path)),
                            'line': line_num,
                            'definition': line.strip()
                        })
        except Exception:
            continue
    
    return results


def find_class(name: str, repo_path: str = None) -> list:
    """Find a class definition by name."""
    results = []
    repo_path = Path(repo_path) if repo_path else Path(__file__).parent.parent.parent
    
    for file_path in repo_path.rglob('*.py'):
        if any(part.startswith('.') or part in ('venv', '__pycache__') 
               for part in file_path.parts):
            continue
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line_num, line in enumerate(f, 1):
                    if f'class {name}' in line:
                        results.append({
                            'file': str(file_path.relative_to(repo_path)),
                            'line': line_num,
                            'definition': line.strip()
                        })
        except Exception:
            continue
    
    return results


def ask_code(question: str) -> str:
    """
    Natural language code search using LLM.
    
    Examples:
        "Where is the function get_llm_provider?"
        "Find the GraphExecutor class"
        "Show me files that use LiteLLMProvider"
    """
    q_lower = question.lower()
    
    # Extract function/class names from question
    words = question.replace('?', '').replace('"', '').replace("'", '').split()
    
    # Try to find function
    if 'function' in q_lower or 'def ' in q_lower:
        for word in words:
            if word not in ('where', 'is', 'the', 'function', 'find', 'show', 'me'):
                results = find_function(word)
                if results:
                    response = f"**Found function `{word}` in:**\n\n"
                    for r in results[:5]:
                        response += f"- `{r['file']}` (line {r['line']})\n"
                        response += f"  ```python\n  {r['definition']}\n  ```\n"
                    return response
    
    # Try to find class
    if 'class' in q_lower:
        for word in words:
            if word not in ('where', 'is', 'the', 'class', 'find', 'show', 'me'):
                results = find_class(word)
                if results:
                    response = f"**Found class `{word}` in:**\n\n"
                    for r in results[:5]:
                        response += f"- `{r['file']}` (line {r['line']})\n"
                        response += f"  ```python\n  {r['definition']}\n  ```\n"
                    return response
    
    # General search
    for word in words:
        if len(word) > 3 and word not in ('where', 'find', 'show', 'what', 'this', 'that'):
            results = search_code(word)
            if results:
                response = f"**Found `{word}` in {len(results)} locations:**\n\n"
                for r in results[:10]:
                    response += f"- `{r['file']}:{r['line']}` - {r['content'][:80]}...\n"
                return response
    
    return "No results found. Try searching for a specific function or class name."


def interactive_mode():
    """Interactive code search."""
    print("=" * 60)
    print("🔍 Code Search - Ask about your codebase")
    print("=" * 60)
    print("\nExamples:")
    print('  - "Where is the function get_llm_provider?"')
    print('  - "Find class GraphExecutor"')
    print('  - "Search for LiteLLMProvider"')
    print("\nType 'quit' to exit.\n")
    
    while True:
        try:
            question = input("\n👤 You: ").strip()
            
            if not question:
                continue
            
            if question.lower() in ('quit', 'exit', 'q'):
                print("\nGoodbye! 👋")
                break
            
            print("\n🔍 ", end="", flush=True)
            answer = ask_code(question)
            print(answer)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break


def main():
    if len(sys.argv) > 1:
        question = ' '.join(sys.argv[1:])
        print(ask_code(question))
    else:
        interactive_mode()


if __name__ == "__main__":
    main()
