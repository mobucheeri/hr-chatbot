"""
hcm_data.py — Mock Oracle Fusion HCM Data Layer
Loads and queries JSON data files that simulate live HCM records.
"""

import json
import os
from typing import Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def _load(filename: str) -> list | dict:
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)


# ── Employee Records ──────────────────────────────────────────────────────────

def get_all_employees() -> list[dict]:
    return _load("employees.json")


def get_employee_by_id(employee_id: str) -> Optional[dict]:
    employees = get_all_employees()
    for emp in employees:
        if emp["id"].upper() == employee_id.upper():
            return emp
    return None


def search_employees_by_name(query: str) -> list[dict]:
    """Case-insensitive name search — matches full name or partial."""
    employees = get_all_employees()
    query_lower = query.lower().strip()
    results = []
    for emp in employees:
        if query_lower in emp["name"].lower():
            results.append(emp)
    return results


def get_employees_by_department(department: str) -> list[dict]:
    employees = get_all_employees()
    dept_lower = department.lower()
    return [e for e in employees if e["department"].lower() == dept_lower]


def get_manager(employee_id: str) -> Optional[dict]:
    emp = get_employee_by_id(employee_id)
    if emp and emp.get("manager_id"):
        return get_employee_by_id(emp["manager_id"])
    return None


def get_direct_reports(manager_id: str) -> list[dict]:
    employees = get_all_employees()
    return [e for e in employees if e.get("manager_id") == manager_id]


# ── Leave Balances ─────────────────────────────────────────────────────────────

def get_all_leave_balances() -> list[dict]:
    return _load("leave_balances.json")


def get_leave_balance(employee_id: str) -> Optional[dict]:
    balances = get_all_leave_balances()
    for record in balances:
        if record["employee_id"].upper() == employee_id.upper():
            return record
    return None


# ── Payslips ───────────────────────────────────────────────────────────────────

def get_all_payslips() -> list[dict]:
    return _load("payslips.json")


def get_payslip(employee_id: str) -> Optional[dict]:
    payslips = get_all_payslips()
    for slip in payslips:
        if slip["employee_id"].upper() == employee_id.upper():
            return slip
    return None


# ── Organisational Structure ──────────────────────────────────────────────────

def get_org_structure() -> dict:
    return _load("org_structure.json")


def get_department_info(department_name: str) -> Optional[dict]:
    org = get_org_structure()
    dept_lower = department_name.lower()
    for dept in org["departments"]:
        if dept["name"].lower() == dept_lower or dept["code"].lower() == dept_lower:
            return dept
    return None


def get_reporting_line(employee_id: str) -> Optional[dict]:
    org = get_org_structure()
    for line in org["reporting_lines"]:
        if line["employee_id"].upper() == employee_id.upper():
            return line
    return None


def get_grade_band(grade: str) -> Optional[dict]:
    org = get_org_structure()
    return org["grade_bands"].get(grade)


# ── Composite Queries ─────────────────────────────────────────────────────────

def get_full_employee_profile(employee_id: str) -> Optional[dict]:
    """Returns employee, their manager's name, and direct report names."""
    emp = get_employee_by_id(employee_id)
    if not emp:
        return None

    profile = dict(emp)

    manager = get_manager(employee_id)
    profile["manager_name"] = manager["name"] if manager else "None (Top of hierarchy)"
    profile["manager_title"] = manager["job_title"] if manager else None

    reports = get_direct_reports(employee_id)
    profile["direct_reports"] = [
        {"id": r["id"], "name": r["name"], "job_title": r["job_title"]}
        for r in reports
    ]

    grade_info = get_grade_band(emp.get("grade", ""))
    profile["grade_band_info"] = grade_info

    return profile


def resolve_employee_from_context(query: str) -> Optional[str]:
    """
    Given a free-text query, attempts to identify an employee ID.
    Returns the employee_id string if a single match is found, else None.
    """
    results = search_employees_by_name(query)
    if len(results) == 1:
        return results[0]["id"]
    if len(results) > 1:
        # Return the closest match (shortest name overlap tends to be most specific)
        for r in results:
            # Prefer exact first/last name match
            name_parts = r["name"].lower().split()
            query_lower = query.lower()
            if any(part == query_lower for part in name_parts):
                return r["id"]
        return results[0]["id"]
    return None
