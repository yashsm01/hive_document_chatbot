# Aden Documentation Chatbot

An AI-powered chatbot that answers questions about Aden using the framework's `LiteLLMProvider`.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   How the Chatbot Works                      │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│   1. SCRAPING (One-time)                                    │
│      docs.adenhq.com ──► scraper.py ──► docs_index.json     │
│                                                              │
│   2. USER QUESTION                                          │
│      "What is Aden?" ──► search.py ──► Find relevant docs   │
│                                                              │
│   3. LLM RESPONSE                                           │
│      [docs context] + [question] ──► LiteLLMProvider ──► Answer │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Files

| File | Description |
|------|-------------|
| `chatbot_llm.py` | Main chatbot with LLM integration |
| `search.py` | Search and find relevant docs |
| `scraper.py` | Scrape docs.adenhq.com |
| `data/docs_index.json` | Indexed documentation |

---

## Quick Start

### Step 1: Set API Key

Add your API key to `.env` file:

```env
# Gemini (recommended)
GEMINI_API_KEY="your-gemini-key"

# Or OpenAI
OPENAI_API_KEY="sk-..."

# Or Anthropic
ANTHROPIC_API_KEY="sk-ant-..."
```

### Step 2: Scrape Documentation (First Time Only)

```bash
cd d:\projects\interview\hive\hive
python docs/chatbot/scraper.py
```

### Step 3: Run Chatbot

```bash
# Interactive mode
python docs/chatbot/chatbot_llm.py

# Single question
python docs/chatbot/chatbot_llm.py "What is Aden?"
```

---

## How It Works

### 1. Document Indexing (`scraper.py`)

Scrapes documentation pages and extracts:
- Page title and URL
- Text content
- Code samples

```python
# Scrape and save docs
python docs/chatbot/scraper.py
# Output: data/docs_index.json
```

### 2. Search (`search.py`)

Finds relevant docs for a question:

```python
from search import search_docs, find_code_samples

# Search for matching docs
results = search_docs("how to create agent", max_results=3)

# Find code examples
codes = find_code_samples("agent.json", max_samples=2)
```

### 3. LLM Integration (`chatbot_llm.py`)

Uses the framework's `LiteLLMProvider`:

```python
from framework.llm import LiteLLMProvider

# Create provider (auto-detects API key)
llm = LiteLLMProvider(model='gemini/gemini-2.5-flash')

# Generate response
response = llm.complete(
    messages=[{"role": "user", "content": question}],
    system="You are the Aden Documentation Assistant...",
    max_tokens=1024
)
```

---

## Example Session

```
============================================================
🤖 Aden Documentation Chatbot
============================================================
✓ Using LLM: gemini/gemini-2.5-flash

Ask me anything about Aden! Type 'quit' to exit.

👤 You: What is Aden?
🤖 Bot: Aden is a platform for building goal-driven AI agents...

👤 You: How to get started?
🤖 Bot: To get started with Aden:
        1. Clone the repository
        2. Run setup script
        3. Set your API key...

👤 You: quit
Goodbye! 👋
```

---

## Supported LLM Providers

| Provider | Model | Env Variable |
|----------|-------|--------------|
| Google | gemini-2.5-flash | `GEMINI_API_KEY` |
| OpenAI | gpt-4o-mini | `OPENAI_API_KEY` |
| Anthropic | claude-3-haiku | `ANTHROPIC_API_KEY` |

---

## Fallback Mode

If no API key is set or LLM fails, the chatbot uses simple keyword-based responses:

```
ℹ️  Running in simple mode (no API key found)
```

Still provides useful responses from indexed documentation!
