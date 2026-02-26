"""
main.py — FastAPI Server
Oracle Fusion HCM AI Assistant — Backend Entry Point

Run with:
    uvicorn backend.main:app --reload --port 8000
"""

import uuid
import logging
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()  # loads .env from the project root
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .prompt_engine import prepare_prompt
from .llm_client import get_llm_response
from . import hcm_data

# ── App Setup ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Meridian HCM AI Assistant",
    description="Oracle Fusion HCM-inspired chatbot backend with LLM integration",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: { session_id: [{"role": ..., "content": ...}, ...] }
sessions: dict[str, list[dict]] = {}


# ── Request / Response Models ──────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    force_provider: Optional[str] = None  # "ollama" | "openai" — for testing


class ChatResponse(BaseModel):
    response: str
    session_id: str
    provider: str
    model: str
    context_keys: list[str]
    intents: list[str]


# ── Health & Info ──────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Meridian HCM AI Assistant"}


# ── HCM Data Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/employees")
async def list_employees(department: Optional[str] = None):
    """Return all employees, optionally filtered by department."""
    if department:
        employees = hcm_data.get_employees_by_department(department)
    else:
        employees = hcm_data.get_all_employees()
    return {"count": len(employees), "employees": employees}


@app.get("/api/employees/{employee_id}")
async def get_employee(employee_id: str):
    """Return full profile for a single employee including manager and reports."""
    profile = hcm_data.get_full_employee_profile(employee_id)
    if not profile:
        raise HTTPException(status_code=404, detail=f"Employee {employee_id} not found")
    return profile


@app.get("/api/leave/{employee_id}")
async def get_leave(employee_id: str):
    """Return leave balance for an employee."""
    leave = hcm_data.get_leave_balance(employee_id)
    if not leave:
        raise HTTPException(status_code=404, detail=f"Leave record for {employee_id} not found")
    return leave


@app.get("/api/payslips/{employee_id}")
async def get_payslip(employee_id: str):
    """Return the most recent payslip for an employee."""
    payslip = hcm_data.get_payslip(employee_id)
    if not payslip:
        raise HTTPException(status_code=404, detail=f"Payslip for {employee_id} not found")
    return payslip


@app.get("/api/org")
async def get_org():
    """Return the full organisational structure."""
    return hcm_data.get_org_structure()


@app.get("/api/org/department/{department_name}")
async def get_department(department_name: str):
    """Return details for a specific department."""
    dept = hcm_data.get_department_info(department_name)
    if not dept:
        raise HTTPException(status_code=404, detail=f"Department '{department_name}' not found")
    return dept


@app.get("/api/search/employees")
async def search_employees(q: str):
    """Search employees by name."""
    results = hcm_data.search_employees_by_name(q)
    return {"count": len(results), "results": results}


# ── Chat Endpoint ──────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    1. Resolves or creates a session.
    2. Detects intent and fetches HCM context.
    3. Builds the prompt and queries the LLM.
    4. Stores the exchange in session history.
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Session management
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = []

    history = sessions[session_id]

    logger.info(f"[{session_id[:8]}] User: {request.message[:80]}")

    # Prompt engineering pipeline
    messages, hcm_context, intents = prepare_prompt(
        user_message=request.message,
        conversation_history=history,
    )

    # LLM call
    response_text, provider, model = await get_llm_response(
        messages=messages,
        force_provider=request.force_provider,
    )

    logger.info(f"[{session_id[:8]}] Provider: {provider} | Intents: {intents}")

    # Update session history
    history.append({"role": "user", "content": request.message})
    history.append({"role": "assistant", "content": response_text})

    # Trim history to avoid unbounded memory growth
    if len(history) > 40:
        sessions[session_id] = history[-40:]

    return ChatResponse(
        response=response_text,
        session_id=session_id,
        provider=provider,
        model=model,
        context_keys=list(hcm_context.keys()),
        intents=intents,
    )


@app.delete("/api/session/{session_id}")
async def clear_session(session_id: str):
    """Clear a chat session's history."""
    if session_id in sessions:
        del sessions[session_id]
    return {"status": "cleared", "session_id": session_id}


# ── Serve Frontend ─────────────────────────────────────────────────────────────

import os

frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")

# Serve all static frontend assets (SVG, CSS, JS files, etc.)
app.mount("/frontend", StaticFiles(directory=frontend_dir), name="frontend")

@app.get("/")
async def serve_frontend():
    return FileResponse(os.path.join(frontend_dir, "index.html"))
