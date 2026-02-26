"""
prompt_engine.py — Prompt Engineering Layer
Detects intent from the user's message, fetches relevant HCM data,
and assembles the final message array sent to the LLM.
"""

import os
import json
import re
from typing import Optional
from . import hcm_data

SYSTEM_PROMPT_PATH = os.path.join(os.path.dirname(__file__), "system_prompt.txt")

# Maximum number of prior turns to include in conversation history
MAX_HISTORY_TURNS = 6


# ── System Prompt ──────────────────────────────────────────────────────────────

def load_system_prompt() -> str:
    """Load the editable system prompt from disk."""
    with open(SYSTEM_PROMPT_PATH, "r") as f:
        return f.read()


# ── Intent Detection ───────────────────────────────────────────────────────────

INTENT_KEYWORDS = {
    "leave": [
        "leave", "holiday", "vacation", "time off", "annual leave", "pto",
        "days off", "absence", "sick leave", "sick day", "booked off",
        "remaining leave", "leave balance", "entitlement",
    ],
    "payslip": [
        "payslip", "pay slip", "salary", "pay", "wage", "wages", "compensation",
        "net pay", "gross", "deduction", "tax", "national insurance", "ni",
        "pension", "take home", "earnings", "income", "paid",
    ],
    "org": [
        "org", "organisation", "organization", "team", "department",
        "reports to", "manager", "hierarchy", "structure", "who manages",
        "direct report", "headcount", "reporting line",
    ],
    "employee": [
        "employee", "profile", "contact", "email", "phone", "location",
        "hire date", "start date", "job title", "role", "grade", "who is",
        "tell me about", "details",
    ],
}


def detect_intents(message: str) -> list[str]:
    """Return a list of intent labels detected from the user message."""
    message_lower = message.lower()
    found = []
    for intent, keywords in INTENT_KEYWORDS.items():
        if any(kw in message_lower for kw in keywords):
            found.append(intent)
    return found if found else ["general"]


# ── Employee Name Extraction ───────────────────────────────────────────────────

def extract_employee_id_from_message(message: str) -> Optional[str]:
    """
    Scans the message for a recognisable employee name or EMP ID.
    Returns the matching employee_id or None.
    """
    # Check for explicit EMP ID pattern
    match = re.search(r'\bEMP\d{3}\b', message, re.IGNORECASE)
    if match:
        return match.group(0).upper()

    # Try matching against known employee names
    employees = hcm_data.get_all_employees()
    message_lower = message.lower()

    best_match = None
    best_score = 0

    for emp in employees:
        name = emp["name"].lower()
        parts = name.split()

        # Full name match (highest priority)
        if name in message_lower:
            return emp["id"]

        # Partial name match — count matching name parts
        score = sum(1 for part in parts if len(part) > 3 and part in message_lower)
        if score > best_score:
            best_score = score
            best_match = emp["id"]

    return best_match if best_score >= 1 else None


# ── HCM Context Assembly ───────────────────────────────────────────────────────

def fetch_hcm_context(intents: list[str], employee_id: Optional[str], message: str) -> dict:
    """
    Given detected intents and a resolved employee ID,
    fetch the relevant HCM data for injection into the prompt.
    """
    context = {}

    if "general" in intents and not employee_id:
        # Generic query — include the employee roster for general context
        context["employee_count"] = len(hcm_data.get_all_employees())
        context["org_summary"] = {
            "company": "Meridian Technologies",
            "offices": ["London", "Edinburgh", "Manchester", "Dublin"],
        }
        return context

    if employee_id:
        # Always include profile for named employee queries
        profile = hcm_data.get_full_employee_profile(employee_id)
        if profile:
            context["employee_profile"] = profile

    if "leave" in intents:
        if employee_id:
            leave = hcm_data.get_leave_balance(employee_id)
            if leave:
                context["leave_balance"] = leave
        else:
            # No employee specified — provide a general leave summary note
            context["leave_note"] = "Leave data requires an employee to be specified."

    if "payslip" in intents:
        if employee_id:
            payslip = hcm_data.get_payslip(employee_id)
            if payslip:
                context["payslip"] = payslip
        else:
            context["payslip_note"] = "Payslip data requires an employee to be specified."

    if "org" in intents:
        org = hcm_data.get_org_structure()
        context["org_structure"] = {
            "departments": org["departments"],
            "grade_bands": org["grade_bands"],
        }
        if employee_id:
            reporting_line = hcm_data.get_reporting_line(employee_id)
            if reporting_line:
                context["reporting_line"] = reporting_line

    if "employee" in intents and not employee_id:
        # List all employees for general employee queries
        employees = hcm_data.get_all_employees()
        context["all_employees"] = [
            {
                "id": e["id"],
                "name": e["name"],
                "department": e["department"],
                "job_title": e["job_title"],
                "location": e["location"],
            }
            for e in employees
        ]

    return context


# ── Message Array Builder ──────────────────────────────────────────────────────

def build_messages(
    user_message: str,
    conversation_history: list[dict],
    hcm_context: dict,
) -> list[dict]:
    """
    Assemble the full message array for the LLM:
    [system prompt] + [trimmed history] + [user message with injected HCM data]
    """
    system_prompt = load_system_prompt()

    # Append current HCM context to the system message for this turn
    if hcm_context:
        context_block = json.dumps(hcm_context, indent=2, default=str)
        system_content = (
            system_prompt
            + "\n\n---\n\n"
            + "## LIVE HCM DATA (current turn)\n\n"
            + "The following data has been retrieved from Oracle Fusion HCM "
            + "and is accurate as of this request. Base your response on this data only.\n\n"
            + f"```json\n{context_block}\n```"
        )
    else:
        system_content = system_prompt

    messages = [{"role": "system", "content": system_content}]

    # Include recent conversation history (trimmed to MAX_HISTORY_TURNS)
    recent_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]
    messages.extend(recent_history)

    # Append the current user message
    messages.append({"role": "user", "content": user_message})

    return messages


# ── Top-Level Orchestration ────────────────────────────────────────────────────

def prepare_prompt(
    user_message: str,
    conversation_history: list[dict],
) -> tuple[list[dict], dict, list[str]]:
    """
    Full pipeline: intent detection → data fetch → message assembly.
    Returns (messages, hcm_context, intents).
    """
    intents = detect_intents(user_message)
    employee_id = extract_employee_id_from_message(user_message)
    hcm_context = fetch_hcm_context(intents, employee_id, user_message)
    messages = build_messages(user_message, conversation_history, hcm_context)

    return messages, hcm_context, intents
