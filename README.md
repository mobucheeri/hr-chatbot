# Meridian HCM AI Assistant

A full-stack HR chatbot demo that simulates an **Oracle Fusion HCM AI assistant**. Built as a portfolio project to demonstrate prompt engineering, LLM integration, RAG-style data injection, and a clean production-ready architecture.

---

## What It Demonstrates

| Concept | Implementation |
|---|---|
| **Prompt Engineering** | Editable `system_prompt.txt` with persona, guardrails, format rules, and tone calibration |
| **RAG-style Context Injection** | Intent detection → HCM data fetch → data injected into LLM context window |
| **LLM Fallback Chain** | Tries local Ollama (Mistral/LLaMA3) first, falls back to OpenAI API |
| **Oracle Fusion HCM Simulation** | REST endpoints mirroring real HCM modules (employees, leave, payroll, org) |
| **Conversation Memory** | Server-side session history with trimming to prevent unbounded growth |
| **Professional Chat UI** | Single-file frontend with markdown rendering, typing indicators, context tags |

---

## Stack

```
Backend:   Python 3.11+ · FastAPI · Uvicorn · httpx
LLM:       Ollama (Mistral / LLaMA3) with OpenAI GPT-3.5 fallback
Frontend:  Vanilla HTML/CSS/JS · marked.js (markdown) · Inter font
Data:      JSON files (employees, leave, payslips, org structure)
```

---

## Project Structure

```
HR CHATBOT/
├── backend/
│   ├── main.py              # FastAPI server — routes, session management
│   ├── hcm_data.py          # Data access layer — loads and queries JSON files
│   ├── llm_client.py        # LLM integration — Ollama first, OpenAI fallback
│   ├── prompt_engine.py     # Intent detection, HCM context fetch, prompt assembly
│   └── system_prompt.txt    # ✏️  Editable AI persona and behavioural rules
├── data/
│   ├── employees.json        # 20 employees with departments, grades, reporting lines
│   ├── leave_balances.json   # Annual leave, sick leave, pending requests per employee
│   ├── payslips.json         # January 2025 payslips with full deduction breakdowns
│   └── org_structure.json    # Departments, reporting hierarchy, grade band definitions
├── frontend/
│   └── index.html            # Complete chat UI — sidebar, messages, typing indicator
├── .env.example              # Environment variable template
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Clone / download and install dependencies

```bash
cd "HR CHATBOT"
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up the LLM

**Option A — Ollama (recommended, runs locally, free)**

```bash
# Install Ollama from https://ollama.com
ollama pull mistral        # or: ollama pull llama3
ollama serve               # starts the local inference server
```

**Option B — OpenAI API (fallback)**

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

The server will **automatically try Ollama first** and fall back to OpenAI if Ollama is not running.

### 3. Start the backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Open the chat UI

Visit **http://localhost:8000** in your browser.

The FastAPI server serves the frontend directly — no separate web server needed.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health check |
| `GET` | `/api/employees` | List all employees (optional `?department=Engineering`) |
| `GET` | `/api/employees/{id}` | Full employee profile with manager and reports |
| `GET` | `/api/leave/{id}` | Leave balance for an employee |
| `GET` | `/api/payslips/{id}` | Latest payslip for an employee |
| `GET` | `/api/org` | Full organisational structure |
| `GET` | `/api/search/employees?q={name}` | Search employees by name |
| `POST` | `/api/chat` | Main chat endpoint |
| `DELETE` | `/api/session/{id}` | Clear a chat session |

### Chat request body

```json
{
  "message": "How many days of leave does Nina Patel have?",
  "session_id": "optional-uuid-for-continuity",
  "force_provider": "openai"
}
```

### Chat response

```json
{
  "response": "Nina Patel currently has **16 days** of annual leave remaining...",
  "session_id": "abc-123",
  "provider": "ollama",
  "model": "mistral",
  "context_keys": ["employee_profile", "leave_balance"],
  "intents": ["employee", "leave"]
}
```

---

## How the Prompt Pipeline Works

```
User message
     │
     ▼
Intent Detection        — keyword matching → ["leave", "employee"]
     │
     ▼
Employee Resolution     — name/ID extraction from free text
     │
     ▼
HCM Data Fetch          — queries the relevant JSON data modules
     │
     ▼
Prompt Assembly         — system_prompt.txt + HCM context + history + user message
     │
     ▼
LLM Call                — Ollama (local) → OpenAI (fallback)
     │
     ▼
Structured Response     — response text + provider metadata + context keys used
```

The HCM context is injected as a JSON block appended to the system prompt each turn. This is a simplified form of **Retrieval-Augmented Generation (RAG)** — instead of a vector database, the context is deterministically selected based on intent detection.

---

## Editing the System Prompt

`backend/system_prompt.txt` controls the AI's persona and behaviour. Edit it freely — changes take effect immediately without restarting the server (it's loaded per-request).

Key sections you can customise:
- **PERSONA** — change the assistant's name and capabilities
- **BEHAVIOURAL GUIDELINES** — adjust how it handles sensitive data
- **RESPONSE FORMAT RULES** — modify output structure and length
- **KNOWLEDGE BOUNDARIES** — define what it can/cannot do
- **COMPANY CONTEXT** — update to reflect a different organisation

---

## Mock Data

The 20 employees at **Meridian Technologies** include:
- Full org hierarchy from CEO down to junior developers
- Realistic UK payroll figures with PAYE, NI, and pension deductions
- Leave balances with varying entitlements (25–30 days depending on grade)
- Offices in London, Edinburgh, Manchester, and Dublin (EUR payroll for Dublin)

To add employees, simply edit the JSON files in `data/` — no code changes required.

---

## Requirements

- Python 3.11+
- Ollama (optional but recommended) OR an OpenAI API key
- Modern web browser
