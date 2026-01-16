# ============================================================================
# Claude Code & MCP Servers Installer
# ============================================================================
# This script installs Claude Code CLI and configures MCP servers for
# Claude Desktop. It handles all prerequisites and lets users choose
# which MCP servers to install.
#
# Author: Ackie
# Version: 2.0.0
# ============================================================================

#Requires -Version 5.1

param(
    [switch]$SkipPrerequisites,
    [switch]$Unattended,
    [switch]$NoRun  # Used when dot-sourcing to import functions without running Main
)

# ============================================================================
# Configuration
# ============================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

# ============================================================================
# Logging Setup
# ============================================================================

$script:LogFile = "$env:USERPROFILE\.claude-mcp-installer.log"

function Write-Log {
    param(
        [string]$Message,
        [string]$Level = "INFO"
    )
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logEntry = "[$timestamp] [$Level] $Message"
    Add-Content -Path $script:LogFile -Value $logEntry -ErrorAction SilentlyContinue
}

function Initialize-Log {
    $separator = "=" * 70
    $header = @"
$separator
CLAUDE CODE & MCP INSTALLER - LOG FILE
Started: $(Get-Date -Format "yyyy-MM-dd HH:mm:ss")
User: $env:USERNAME
Computer: $env:COMPUTERNAME
OS: $([System.Environment]::OSVersion.VersionString)
PowerShell: $($PSVersionTable.PSVersion)
$separator
"@
    Set-Content -Path $script:LogFile -Value $header -ErrorAction SilentlyContinue
    Write-Log "Installer started"
    Write-Log "Log file: $script:LogFile"
}

# Colors for output
$colors = @{
    Header = "Cyan"
    Success = "Green"
    Warning = "Yellow"
    Error = "Red"
    Info = "White"
    Menu = "Magenta"
    Installed = "DarkGreen"
    NotInstalled = "DarkRed"
}

# Global status tracking
$script:InstalledComponents = @{
    NodeJS = $false
    Python = $false
    UV = $false
    ClaudeCode = $false
}

# ============================================================================
# Embedded Conversation Watchdog MCP Script
# ============================================================================

$WatchdogScript = @'
#!/usr/bin/env python3
"""
Conversation Watchdog MCP Server
================================
An MCP server that monitors Claude Desktop conversations for truncation,
tracks progress, and enables automatic recovery from failed/incomplete responses.

Author: Built for QBD Group
Version: 1.0.0
"""

import json
import os
import re
import hashlib
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any
from enum import Enum
from dataclasses import dataclass, asdict
import asyncio

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict


# Initialize MCP Server
mcp = FastMCP("conversation_watchdog_mcp")

# Configuration
WATCHDOG_DATA_DIR = Path.home() / ".conversation_watchdog"
TASKS_FILE = WATCHDOG_DATA_DIR / "tasks.json"
HISTORY_FILE = WATCHDOG_DATA_DIR / "history.json"
CONFIG_FILE = WATCHDOG_DATA_DIR / "config.json"
AUTO_START_FILE = WATCHDOG_DATA_DIR / "auto_start.json"
LOG_FILE = WATCHDOG_DATA_DIR / "watchdog.log"

# Ensure data directory exists
WATCHDOG_DATA_DIR.mkdir(exist_ok=True)

# Auto-start configuration - ALWAYS ON
AUTO_START_ENABLED = True


def setup_logging():
    """Configure logging with file and console output."""
    logger = logging.getLogger("conversation_watchdog")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    return logger

log = setup_logging()
log.info("=" * 60)
log.info("WATCHDOG MCP SERVER STARTING")
log.info(f"Data directory: {WATCHDOG_DATA_DIR}")
log.info(f"Log file: {LOG_FILE}")
log.info(f"Auto-start enabled: {AUTO_START_ENABLED}")
log.info("=" * 60)


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    NEEDS_CONTINUATION = "needs_continuation"


class TruncationType(str, Enum):
    MID_SENTENCE = "mid_sentence"
    MID_WORD = "mid_word"
    MID_CODE_BLOCK = "mid_code_block"
    MID_LIST = "mid_list"
    INCOMPLETE_THOUGHT = "incomplete_thought"
    CLEAN_END = "clean_end"
    UNKNOWN = "unknown"


@dataclass
class TrackedTask:
    task_id: str
    original_question: str
    status: TaskStatus
    created_at: str
    updated_at: str
    checkpoints: List[Dict[str, Any]]
    last_response_snippet: Optional[str] = None
    truncation_type: Optional[TruncationType] = None
    retry_count: int = 0
    reformulations: List[str] = None
    completion_indicators: List[str] = None

    def __post_init__(self):
        if self.reformulations is None:
            self.reformulations = []
        if self.completion_indicators is None:
            self.completion_indicators = []


def load_tasks() -> Dict[str, TrackedTask]:
    log.debug(f"Loading tasks from {TASKS_FILE}")
    if not TASKS_FILE.exists():
        return {}
    try:
        with open(TASKS_FILE, 'r') as f:
            data = json.load(f)
            return {k: TrackedTask(**v) for k, v in data.items()}
    except (json.JSONDecodeError, TypeError) as e:
        log.error(f"Error loading tasks: {e}")
        return {}


def save_tasks(tasks: Dict[str, TrackedTask]) -> None:
    with open(TASKS_FILE, 'w') as f:
        json.dump({k: asdict(v) for k, v in tasks.items()}, f, indent=2)
    log.info(f"Saved {len(tasks)} tasks to storage")


def load_history() -> List[Dict[str, Any]]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-100:], f, indent=2)


def generate_task_id(question: str) -> str:
    content = f"{question}_{datetime.now().isoformat()}"
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def get_or_create_auto_task(question: str = None) -> Optional[TrackedTask]:
    tasks = load_tasks()
    in_progress = [t for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS]
    if in_progress:
        return max(in_progress, key=lambda t: t.updated_at)
    if AUTO_START_ENABLED:
        task_id = generate_task_id(question or "Auto-tracked conversation")
        now = datetime.now().isoformat()
        task = TrackedTask(
            task_id=task_id,
            original_question=question or "Auto-tracked conversation - watchdog always active",
            status=TaskStatus.IN_PROGRESS,
            created_at=now,
            updated_at=now,
            checkpoints=[{"timestamp": now, "description": "Auto-started by watchdog", "completion_percentage": 0, "checkpoint_number": 1}]
        )
        tasks[task_id] = task
        save_tasks(tasks)
        log.info(f"AUTO-CREATED new task: {task_id}")
        return task
    return None


def ensure_watchdog_active() -> TrackedTask:
    return get_or_create_auto_task()


TRUNCATION_PATTERNS = {
    TruncationType.MID_SENTENCE: [r'[a-zA-Z0-9]\s*$', r',\s*$', r':\s*$', r'\band\s*$', r'\bor\s*$', r'\bthe\s*$', r'\ba\s*$', r'\bto\s*$', r'\bfor\s*$', r'\bwith\s*$'],
    TruncationType.MID_WORD: [r'[a-zA-Z]{2,}$'],
    TruncationType.MID_CODE_BLOCK: [r'```[a-zA-Z]*\n(?!.*```)', r'`[^`]+$', r'\{[^}]*$', r'\[[^\]]*$', r'\([^)]*$'],
    TruncationType.MID_LIST: [r'\n\d+\.\s*$', r'\n[-*]\s*$', r'\n\d+\.\s+\w+[^.!?]*$'],
    TruncationType.INCOMPLETE_THOUGHT: [r'(?:First|Second|Third|Fourth|Fifth|1\)|2\)|3\)|Step \d)[^.!?]*$', r'(?:However|Therefore|Additionally|Furthermore|Moreover)[^.!?]*$', r'(?:For example|Such as|Including)[^.!?]*$'],
}

CLEAN_END_PATTERNS = [r'[.!?]\s*$', r'```\s*$', r'\n\s*$', r'(?:Let me know|Hope this helps|Feel free|Good luck)[^.]*[.!]\s*$']


def detect_truncation(text: str) -> TruncationType:
    if not text or len(text.strip()) < 10:
        return TruncationType.UNKNOWN
    text = text.strip()
    for pattern in CLEAN_END_PATTERNS:
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return TruncationType.CLEAN_END
    for truncation_type, patterns in TRUNCATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return truncation_type
    lines = text.split('\n')
    last_line = lines[-1].strip() if lines else ""
    if last_line and not re.search(r'[.!?:;]$', last_line) and len(last_line) > 20:
        return TruncationType.MID_SENTENCE
    if text.count('```') % 2 != 0:
        return TruncationType.MID_CODE_BLOCK
    return TruncationType.UNKNOWN


def calculate_completion_confidence(text: str) -> float:
    if not text:
        return 0.0
    score = 0.5
    text = text.strip()
    if re.search(r'[.!?]\s*$', text):
        score += 0.15
    if re.search(r'(?:Let me know|Hope this helps|Feel free|Good luck)', text, re.IGNORECASE):
        score += 0.15
    if text.count('```') % 2 == 0:
        score += 0.1
    if len(text) > 500:
        score += 0.05
    truncation = detect_truncation(text)
    if truncation == TruncationType.MID_SENTENCE:
        score -= 0.3
    elif truncation == TruncationType.MID_WORD:
        score -= 0.4
    elif truncation == TruncationType.MID_CODE_BLOCK:
        score -= 0.35
    elif truncation == TruncationType.MID_LIST:
        score -= 0.25
    elif truncation == TruncationType.INCOMPLETE_THOUGHT:
        score -= 0.2
    return max(0.0, min(1.0, score))


REFORMULATION_STRATEGIES = [
    {"name": "simplify", "description": "Break down into smaller parts", "template": "Let me approach this differently. Instead of answering everything at once, let's focus on the most important part first:\n\n{focus_question}\n\nPlease provide a complete answer to just this part."},
    {"name": "chunk", "description": "Request chunked responses", "template": "Please answer this question in clearly marked parts. After each part, pause and ask if I want you to continue.\n\nQuestion: {question}\n\nStart with Part 1."},
    {"name": "outline_first", "description": "Get outline then expand", "template": "For this question, please first provide a brief outline of your answer (just the main points), then I'll ask you to expand on specific parts:\n\n{question}"},
    {"name": "specific_format", "description": "Request specific format", "template": "Please answer the following question. Keep your response concise and complete. If you need more space, end with '[CONTINUED]' and I'll ask for the rest.\n\n{question}"},
    {"name": "reverse_approach", "description": "Ask for conclusion first", "template": "For this question, please start with your conclusion/answer first, then provide supporting details:\n\n{question}"},
]


def generate_reformulation(original_question: str, previous_attempts: List[str], truncation_type: TruncationType) -> Dict[str, str]:
    used_strategies = set()
    for attempt in previous_attempts:
        for strategy in REFORMULATION_STRATEGIES:
            if strategy["name"] in attempt:
                used_strategies.add(strategy["name"])
    strategy = None
    if truncation_type == TruncationType.MID_CODE_BLOCK:
        preferred = ["chunk", "outline_first"]
    elif truncation_type == TruncationType.MID_LIST:
        preferred = ["chunk", "specific_format"]
    elif truncation_type in [TruncationType.MID_SENTENCE, TruncationType.MID_WORD]:
        preferred = ["simplify", "chunk", "specific_format"]
    else:
        preferred = ["outline_first", "simplify", "reverse_approach"]
    for pref in preferred:
        if pref not in used_strategies:
            strategy = next((s for s in REFORMULATION_STRATEGIES if s["name"] == pref), None)
            if strategy:
                break
    if not strategy:
        for s in REFORMULATION_STRATEGIES:
            if s["name"] not in used_strategies:
                strategy = s
                break
    if not strategy:
        strategy = REFORMULATION_STRATEGIES[0]
    focus_question = original_question
    if "focus_question" in strategy["template"]:
        if "?" in original_question:
            focus_question = original_question.split("?")[0] + "?"
        else:
            words = original_question.split()
            focus_question = " ".join(words[:min(20, len(words))]) + "..."
    reformulated = strategy["template"].format(question=original_question, focus_question=focus_question)
    return {"strategy": strategy["name"], "reformulated_question": reformulated, "explanation": strategy["description"]}


class StartTaskInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    question: str = Field(..., description="The original question/task being worked on", min_length=5, max_length=10000)
    expected_completion_indicators: Optional[List[str]] = Field(default=None, description="Optional phrases that indicate completion")


class CheckpointInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    task_id: str = Field(..., description="The task ID to checkpoint")
    progress_description: str = Field(..., description="Description of current progress", min_length=1, max_length=5000)
    completion_percentage: Optional[int] = Field(default=None, description="Estimated completion percentage (0-100)", ge=0, le=100)


class CheckCompletionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    task_id: Optional[str] = Field(default=None, description="Specific task ID to check, or None for the most recent")
    response_text: str = Field(..., description="The response text to analyze for completeness", min_length=1)


class GetRecoveryPlanInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    task_id: Optional[str] = Field(default=None, description="Specific task ID, or None for most recent incomplete task")


class MarkCompleteInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    task_id: str = Field(..., description="The task ID to mark complete")
    final_notes: Optional[str] = Field(default=None, description="Optional notes about the completion")


class ListTasksInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    status_filter: Optional[TaskStatus] = Field(default=None, description="Filter by status")
    limit: int = Field(default=10, description="Maximum number of tasks to return", ge=1, le=50)


class AutoStartInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')
    question: Optional[str] = Field(default=None, description="The question/task being worked on (optional)")


@mcp.tool(name="watchdog_auto_activate", annotations={"title": "Auto-Activate Watchdog", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def auto_activate(params: AutoStartInput = None) -> str:
    log.info("TOOL CALLED: watchdog_auto_activate")
    question = params.question if params else None
    task = get_or_create_auto_task(question)
    if task:
        return json.dumps({"success": True, "watchdog_status": "ACTIVE", "mode": "always_on", "task_id": task.task_id, "message": "Watchdog is now monitoring this conversation."}, indent=2)
    return json.dumps({"success": False, "error": "Could not activate watchdog"}, indent=2)


@mcp.tool(name="watchdog_start_task", annotations={"title": "Start Tracking a Task", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": False})
async def start_task(params: StartTaskInput) -> str:
    log.info(f"TOOL CALLED: watchdog_start_task - {params.question[:100]}...")
    tasks = load_tasks()
    task_id = generate_task_id(params.question)
    now = datetime.now().isoformat()
    task = TrackedTask(task_id=task_id, original_question=params.question, status=TaskStatus.IN_PROGRESS, created_at=now, updated_at=now, checkpoints=[], completion_indicators=params.expected_completion_indicators or [])
    tasks[task_id] = task
    save_tasks(tasks)
    return json.dumps({"success": True, "task_id": task_id, "message": f"Now tracking task. Use task_id '{task_id}' for checkpoints."}, indent=2)


@mcp.tool(name="watchdog_checkpoint", annotations={"title": "Save Progress Checkpoint", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def save_checkpoint(params: CheckpointInput) -> str:
    log.info(f"TOOL CALLED: watchdog_checkpoint - Task: {params.task_id}")
    tasks = load_tasks()
    if params.task_id not in tasks:
        return json.dumps({"success": False, "error": f"Task '{params.task_id}' not found"}, indent=2)
    task = tasks[params.task_id]
    now = datetime.now().isoformat()
    checkpoint = {"timestamp": now, "description": params.progress_description, "completion_percentage": params.completion_percentage, "checkpoint_number": len(task.checkpoints) + 1}
    task.checkpoints.append(checkpoint)
    task.updated_at = now
    tasks[params.task_id] = task
    save_tasks(tasks)
    return json.dumps({"success": True, "checkpoint_number": checkpoint["checkpoint_number"], "message": "Progress saved."}, indent=2)


@mcp.tool(name="watchdog_check_completion", annotations={"title": "Check Response Completeness", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def check_completion(params: CheckCompletionInput) -> str:
    log.info("TOOL CALLED: watchdog_check_completion")
    ensure_watchdog_active()
    tasks = load_tasks()
    task = None
    task_id = params.task_id
    if task_id and task_id in tasks:
        task = tasks[task_id]
    elif not task_id:
        in_progress = [t for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        if in_progress:
            task = max(in_progress, key=lambda t: t.updated_at)
            task_id = task.task_id
    truncation_type = detect_truncation(params.response_text)
    confidence = calculate_completion_confidence(params.response_text)
    is_complete = truncation_type == TruncationType.CLEAN_END and confidence >= 0.7
    if task:
        now = datetime.now().isoformat()
        task.last_response_snippet = params.response_text[-500:] if len(params.response_text) > 500 else params.response_text
        task.truncation_type = truncation_type
        task.updated_at = now
        if is_complete:
            task.status = TaskStatus.COMPLETED
        elif truncation_type != TruncationType.CLEAN_END:
            task.status = TaskStatus.TRUNCATED
        tasks[task_id] = task
        save_tasks(tasks)
    result = {"is_complete": is_complete, "confidence": round(confidence, 2), "truncation_type": truncation_type.value, "analysis": {"response_length": len(params.response_text), "ends_with_punctuation": bool(re.search(r'[.!?]\s*$', params.response_text.strip())), "has_balanced_code_blocks": params.response_text.count('```') % 2 == 0}}
    if not is_complete:
        result["recommendation"] = "Use watchdog_get_recovery_plan to get continuation strategy"
    if task_id:
        result["task_id"] = task_id
    return json.dumps(result, indent=2)


@mcp.tool(name="watchdog_get_recovery_plan", annotations={"title": "Get Recovery Plan", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def get_recovery_plan(params: GetRecoveryPlanInput) -> str:
    log.info("TOOL CALLED: watchdog_get_recovery_plan")
    ensure_watchdog_active()
    tasks = load_tasks()
    task = None
    if params.task_id and params.task_id in tasks:
        task = tasks[params.task_id]
    else:
        incomplete = [t for t in tasks.values() if t.status in [TaskStatus.TRUNCATED, TaskStatus.NEEDS_CONTINUATION, TaskStatus.IN_PROGRESS]]
        if incomplete:
            task = max(incomplete, key=lambda t: t.updated_at)
    if not task:
        return json.dumps({"success": False, "error": "No incomplete tasks found."}, indent=2)
    truncation_type = task.truncation_type or TruncationType.UNKNOWN
    reformulation = generate_reformulation(task.original_question, task.reformulations, truncation_type)
    continuation_context = ""
    if task.checkpoints:
        last_checkpoint = task.checkpoints[-1]
        continuation_context = f"\n\nLast checkpoint ({last_checkpoint['timestamp']}):\n{last_checkpoint['description']}"
    recovery_plan = {"task_id": task.task_id, "original_question": task.original_question, "truncation_type": truncation_type.value if truncation_type else "unknown", "retry_count": task.retry_count, "recovery_strategy": {"name": reformulation["strategy"], "explanation": reformulation["explanation"], "reformulated_question": reformulation["reformulated_question"]}, "last_progress": continuation_context if continuation_context else "No checkpoints saved"}
    task.reformulations.append(f"{reformulation['strategy']}: {reformulation['reformulated_question'][:100]}...")
    task.retry_count += 1
    task.status = TaskStatus.NEEDS_CONTINUATION
    task.updated_at = datetime.now().isoformat()
    tasks[task.task_id] = task
    save_tasks(tasks)
    return json.dumps(recovery_plan, indent=2)


@mcp.tool(name="watchdog_mark_complete", annotations={"title": "Mark Task Complete", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def mark_complete(params: MarkCompleteInput) -> str:
    log.info(f"TOOL CALLED: watchdog_mark_complete - Task: {params.task_id}")
    tasks = load_tasks()
    if params.task_id not in tasks:
        return json.dumps({"success": False, "error": f"Task '{params.task_id}' not found"}, indent=2)
    task = tasks[params.task_id]
    now = datetime.now().isoformat()
    task.status = TaskStatus.COMPLETED
    task.updated_at = now
    if params.final_notes:
        task.checkpoints.append({"timestamp": now, "description": f"COMPLETED: {params.final_notes}", "completion_percentage": 100})
    tasks[params.task_id] = task
    save_tasks(tasks)
    history = load_history()
    history.append({"task_id": task.task_id, "question_preview": task.original_question[:100] + "...", "status": "completed", "retry_count": task.retry_count, "completed_at": now})
    save_history(history)
    return json.dumps({"success": True, "message": f"Task '{params.task_id}' marked as complete", "retry_count": task.retry_count}, indent=2)


@mcp.tool(name="watchdog_list_tasks", annotations={"title": "List Tracked Tasks", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def list_tasks(params: ListTasksInput) -> str:
    log.info("TOOL CALLED: watchdog_list_tasks")
    ensure_watchdog_active()
    tasks = load_tasks()
    filtered = list(tasks.values())
    if params.status_filter:
        filtered = [t for t in filtered if t.status == params.status_filter]
    filtered.sort(key=lambda t: t.updated_at, reverse=True)
    filtered = filtered[:params.limit]
    task_list = [{"task_id": task.task_id, "status": task.status.value, "question_preview": task.original_question[:80] + "..." if len(task.original_question) > 80 else task.original_question, "created_at": task.created_at, "updated_at": task.updated_at, "checkpoint_count": len(task.checkpoints), "retry_count": task.retry_count} for task in filtered]
    return json.dumps({"total_tasks": len(tasks), "filtered_count": len(task_list), "tasks": task_list}, indent=2)


@mcp.tool(name="watchdog_analyze_text", annotations={"title": "Analyze Text for Truncation", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def analyze_text(response_text: str) -> str:
    log.info("TOOL CALLED: watchdog_analyze_text")
    ensure_watchdog_active()
    truncation_type = detect_truncation(response_text)
    confidence = calculate_completion_confidence(response_text)
    is_complete = truncation_type == TruncationType.CLEAN_END and confidence >= 0.7
    return json.dumps({"truncation_type": truncation_type.value, "completion_confidence": round(confidence, 2), "is_likely_complete": is_complete, "text_length": len(response_text), "analysis_details": {"ends_properly": bool(re.search(r'[.!?]\s*$', response_text.strip())), "balanced_code_blocks": response_text.count('```') % 2 == 0, "balanced_brackets": response_text.count('[') == response_text.count(']'), "balanced_braces": response_text.count('{') == response_text.count('}')}}, indent=2)


@mcp.tool(name="watchdog_get_status", annotations={"title": "Get Watchdog Status", "readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": False})
async def get_status() -> str:
    log.info("TOOL CALLED: watchdog_get_status")
    active_task = ensure_watchdog_active()
    tasks = load_tasks()
    history = load_history()
    status_counts = {}
    for task in tasks.values():
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1
    needs_attention = [{"task_id": t.task_id, "status": t.status.value, "question_preview": t.original_question[:50] + "..."} for t in tasks.values() if t.status in [TaskStatus.TRUNCATED, TaskStatus.NEEDS_CONTINUATION, TaskStatus.FAILED]]
    return json.dumps({"system_status": "ACTIVE - ALWAYS ON", "always_on_mode": True, "auto_start_enabled": AUTO_START_ENABLED, "current_active_task": active_task.task_id if active_task else None, "data_directory": str(WATCHDOG_DATA_DIR), "log_file": str(LOG_FILE), "statistics": {"total_tracked_tasks": len(tasks), "completed_in_history": len([h for h in history if h.get("status") == "completed"]), "status_breakdown": status_counts}, "needs_attention": needs_attention[:5]}, indent=2)


if __name__ == "__main__":
    mcp.run()
'@

# MCP Server definitions
$MCPServers = @{
    "powerbi-modeling-mcp" = @{
        Name = "PowerBI Modeling MCP"
        Description = "Power BI semantic model manipulation via Tabular Editor"
        Type = "executable"
        RequiresUsername = $true
        UsernamePrompt = "Enter your Windows username (the only part that changes between environments)"
        UsernameDefault = "$env:USERNAME"
        PathTemplate = "C:\Users\{USERNAME}\.vscode\extensions"
        PathPattern = "analysis-services.powerbi-modeling-mcp-*-win32-x64"
        Prerequisites = @()
        Config = @{
            command = "{PATH}\server\powerbi-modeling-mcp.exe"
            args = @("--start")
        }
    }
    "ms365" = @{
        Name = "Microsoft 365 MCP"
        Description = "Access Microsoft 365 services (Outlook, OneDrive, Calendar, Teams, etc.)"
        Type = "npx"
        RequiresPath = $false
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@softeria/ms-365-mcp-server", "--org-mode")
        }
    }
    "teams-mcp" = @{
        Name = "Teams MCP"
        Description = "Microsoft Teams integration for chats and channels"
        Type = "npx"
        RequiresPath = $false
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@floriscornel/teams-mcp@latest")
        }
    }
    "excel" = @{
        Name = "Excel MCP"
        Description = "Read and write Excel files, create tables and charts"
        Type = "npx"
        RequiresPath = $false
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("--yes", "@negokaz/excel-mcp-server")
            env = @{
                EXCEL_MCP_PAGING_CELLS_LIMIT = "4000"
            }
        }
    }
    "word-document-server" = @{
        Name = "Word Document MCP"
        Description = "Create and edit Word documents"
        Type = "uvx"
        RequiresPath = $false
        Prerequisites = @("python", "uv")
        Config = @{
            command = "uvx"
            args = @("--from", "office-word-mcp-server", "word_mcp_server")
        }
    }
    "powerpoint" = @{
        Name = "PowerPoint MCP"
        Description = "Create and edit PowerPoint presentations"
        Type = "uvx"
        RequiresPath = $false
        Prerequisites = @("python", "uv")
        Config = @{
            command = "uvx"
            args = @("--from", "office-powerpoint-mcp-server", "ppt_mcp_server")
        }
    }
    "teamtailor" = @{
        Name = "TeamTailor MCP"
        Description = "TeamTailor recruitment platform integration"
        Type = "npx"
        RequiresPath = $false
        RequiresApiKey = $true
        ApiKeyPrompt = "Enter your TeamTailor API key"
        ApiKeyHelp = @"

  Where to find your TeamTailor API key:
  1. Log in to TeamTailor (https://app.teamtailor.com)
  2. Go to Settings (gear icon) > Integrations > API keys
  3. Click 'New API key' or copy an existing one
  4. Make sure the key has read access to candidates

"@
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@crunchloop/mcp-teamtailor")
            env = @{
                TEAMTAILOR_URL = "https://api.teamtailor.com/v1"
                TEAMTAILOR_API_KEY = "{API_KEY}"
            }
        }
    }
    "conversation-watchdog" = @{
        Name = "Conversation Watchdog MCP"
        Description = "Monitors conversations for truncation and enables recovery"
        Type = "embedded-python"
        RequiresPath = $false
        Prerequisites = @("python")
        PythonPackages = @("mcp[cli]", "pydantic", "fastmcp")
        Config = @{
            command = "py"
            args = @("{WATCHDOG_PATH}")
        }
    }
    "netsuite" = @{
        Name = "NetSuite MCP"
        Description = "Oracle NetSuite ERP integration with SuiteQL and REST API access"
        Type = "npx"
        RequiresPath = $false
        RequiresApiKey = $true
        ApiKeyPrompt = "Enter your NetSuite Account ID (e.g., 1234567 or 1234567-sb1 for sandbox)"
        ApiKeyHelp = @"

  NetSuite MCP Setup - OAuth 2.0 Configuration Required:

  Before using this MCP, you need to set up OAuth 2.0 in NetSuite:
  1. Go to Setup > Integration > Manage Integrations > New
  2. Enable 'Token-Based Authentication' and 'OAuth 2.0'
  3. Set the callback URL to: http://localhost:9000/callback
  4. Copy the Client ID and Client Secret
  5. Set up a Role with appropriate permissions for API access

  Environment variables needed (set in your system):
  - NETSUITE_ACCOUNT_ID: Your account ID (entered above)
  - NETSUITE_CLIENT_ID: OAuth 2.0 Client ID
  - NETSUITE_CLIENT_SECRET: OAuth 2.0 Client Secret

  For detailed setup: https://github.com/suiteinsider/netsuite-mcp

"@
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@anthropic/mcp-server-netsuite")
            env = @{
                NETSUITE_ACCOUNT_ID = "{API_KEY}"
            }
        }
    }
    "google-drive" = @{
        Name = "Google Drive MCP"
        Description = "Google Drive, Docs, Sheets, and Slides integration"
        Type = "npx"
        RequiresPath = $true
        PathPrompt = "Enter the path to your Google OAuth credentials JSON file"
        PathDefault = "$env:USERPROFILE\.google\credentials.json"
        PathHelp = @"

  Google Drive MCP Setup - OAuth Credentials Required:

  1. Go to Google Cloud Console (https://console.cloud.google.com)
  2. Create a new project or select existing one
  3. Enable the following APIs:
     - Google Drive API
     - Google Docs API
     - Google Sheets API
     - Google Slides API
  4. Go to Credentials > Create Credentials > OAuth 2.0 Client ID
  5. Select 'Desktop application' as the application type
  6. Download the JSON credentials file
  7. Save it to the path you enter above

  On first run, the MCP will open a browser for authentication.

"@
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@anthropic/mcp-server-gdrive")
            env = @{
                GDRIVE_CREDENTIALS_PATH = "{PATH}"
            }
        }
    }
    "google-cloud-storage" = @{
        Name = "Google Cloud Storage MCP"
        Description = "Manage Google Cloud Storage buckets and objects"
        Type = "npx"
        RequiresPath = $false
        RequiresApiKey = $true
        ApiKeyPrompt = "Enter your Google Cloud project ID"
        ApiKeyHelp = @"

  Google Cloud Storage MCP Setup:

  Prerequisites:
  1. Install Google Cloud SDK (gcloud CLI)
  2. Run: gcloud auth application-default login
  3. Your project ID can be found in Google Cloud Console

  The MCP uses Application Default Credentials (ADC) for authentication.
  Make sure you have the Storage Admin role or appropriate permissions.

  For more info: https://www.npmjs.com/package/@google-cloud/storage-mcp

"@
        Prerequisites = @("nodejs")
        Config = @{
            command = "npx"
            args = @("-y", "@google-cloud/storage-mcp")
            env = @{
                GOOGLE_CLOUD_PROJECT = "{API_KEY}"
            }
        }
    }
    "cli-microsoft365" = @{
        Name = "CLI for Microsoft 365 MCP"
        Description = "Manage Microsoft 365 using PnP CLI - SharePoint, Teams, Entra ID, Power Platform and more"
        Type = "npx"
        RequiresPath = $false
        Prerequisites = @("nodejs")
        PreInstallNote = @"

  CLI for Microsoft 365 MCP Server - Additional Setup Required:

  This MCP server requires the CLI for Microsoft 365 to be installed globally:
  1. Run: npm i -g @pnp/cli-microsoft365
  2. Configure the CLI:
     m365 cli config set --key prompt --value false
     m365 cli config set --key output --value text
     m365 cli config set --key helpMode --value full
  3. Authenticate: m365 login

  The MCP server will use your existing CLI authentication context.
  For more info: https://github.com/pnp/cli-microsoft365-mcp-server

"@
        Config = @{
            command = "npx"
            args = @("-y", "@pnp/cli-microsoft365-mcp-server@latest")
        }
    }
    "notion" = @{
        Name = "Notion MCP"
        Description = "Official Notion MCP Server - Access and manage Notion workspaces, pages, databases and blocks"
        Type = "npx"
        RequiresPath = $false
        RequiresApiKey = $true
        ApiKeyUrl = "https://www.notion.so/profile/integrations"
        Prerequisites = @("nodejs")
        PreInstallNote = @"

  Notion MCP Server - Setup Required:

  1. Create a Notion integration at https://www.notion.so/profile/integrations
  2. Copy the Internal Integration Secret (starts with 'ntn_' or 'secret_')
  3. Share the pages/databases you want to access with your integration

  For more info: https://developers.notion.com/docs/mcp

"@
        Config = @{
            command = "npx"
            args = @("-y", "@notionhq/notion-mcp-server")
            env = @{
                NOTION_TOKEN = "{API_KEY}"
            }
        }
    }
    "airtable" = @{
        Name = "Airtable MCP"
        Description = "Airtable MCP Server - Read and write to Airtable bases, tables and records"
        Type = "npx"
        RequiresPath = $false
        RequiresApiKey = $true
        ApiKeyUrl = "https://airtable.com/create/tokens"
        Prerequisites = @("nodejs")
        PreInstallNote = @"

  Airtable MCP Server - Setup Required:

  1. Create a Personal Access Token at https://airtable.com/create/tokens
  2. Grant scopes: schema.bases:read, data.records:read (and write scopes if needed)
  3. Select the bases you want to access

  For more info: https://github.com/domdomegg/airtable-mcp-server

"@
        Config = @{
            command = "npx"
            args = @("-y", "airtable-mcp-server")
            env = @{
                AIRTABLE_API_KEY = "{API_KEY}"
            }
        }
    }
}

# ============================================================================
# Helper Functions
# ============================================================================

function Write-Header {
    param([string]$Text)
    Write-Host ""
    Write-Host ("=" * 70) -ForegroundColor $colors.Header
    Write-Host "  $Text" -ForegroundColor $colors.Header
    Write-Host ("=" * 70) -ForegroundColor $colors.Header
    Write-Host ""
}

function Write-Step {
    param([string]$Text)
    Write-Host "[*] $Text" -ForegroundColor $colors.Info
    Write-Log $Text "STEP"
}

function Write-Success {
    param([string]$Text)
    Write-Host "[OK] $Text" -ForegroundColor $colors.Success
    Write-Log $Text "OK"
}

function Write-Warn {
    param([string]$Text)
    Write-Host "[!] $Text" -ForegroundColor $colors.Warning
    Write-Log $Text "WARN"
}

function Write-Err {
    param([string]$Text)
    Write-Host "[X] $Text" -ForegroundColor $colors.Error
    Write-Log $Text "ERROR"
}

function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Test-CommandExists {
    param([string]$Command)
    $oldPreference = $ErrorActionPreference
    $ErrorActionPreference = "SilentlyContinue"
    try {
        if (Get-Command $Command -ErrorAction SilentlyContinue) { return $true }
    } catch { }
    $ErrorActionPreference = $oldPreference
    return $false
}

function Refresh-Path {
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" +
                [System.Environment]::GetEnvironmentVariable("Path", "User")
}

# ============================================================================
# Component Detection Functions
# ============================================================================

function Get-NodeVersion {
    Write-Log "Checking for Node.js..." "DEBUG"
    if (Test-CommandExists "node") {
        try {
            $version = node --version 2>$null
            Write-Log "Node.js found: $version" "DEBUG"
            return $version
        } catch {
            Write-Log "Node.js command failed: $_" "DEBUG"
        }
    }
    Write-Log "Node.js not found" "DEBUG"
    return $null
}

function Get-PythonVersion {
    Write-Log "Checking for Python..." "DEBUG"
    if (Test-CommandExists "py") {
        try {
            $version = py --version 2>$null
            Write-Log "Python (py) found: $version" "DEBUG"
            return $version
        } catch {
            Write-Log "Python (py) command failed: $_" "DEBUG"
        }
    }
    if (Test-CommandExists "python") {
        try {
            $version = python --version 2>$null
            if ($version -match "Python 3") {
                Write-Log "Python found: $version" "DEBUG"
                return $version
            }
        } catch {
            Write-Log "Python command failed: $_" "DEBUG"
        }
    }
    Write-Log "Python not found" "DEBUG"
    return $null
}

function Get-UVVersion {
    Write-Log "Checking for UV..." "DEBUG"
    if (Test-CommandExists "uvx") {
        try {
            $version = uv --version 2>$null
            Write-Log "UV found: $version" "DEBUG"
            return $version
        } catch {
            Write-Log "UV command failed: $_" "DEBUG"
        }
    }
    Write-Log "UV not found" "DEBUG"
    return $null
}

function Get-ClaudeVersion {
    Write-Log "Checking for Claude Code CLI..." "DEBUG"
    if (Test-CommandExists "claude") {
        try {
            $version = claude --version 2>$null
            Write-Log "Claude Code CLI found: $version" "DEBUG"
            return $version
        } catch {
            Write-Log "Claude Code CLI command failed: $_" "DEBUG"
        }
    }
    Write-Log "Claude Code CLI not found" "DEBUG"
    return $null
}

function Show-ComponentStatus {
    Write-Header "System Component Status"

    # Check Node.js
    $nodeVersion = Get-NodeVersion
    if ($nodeVersion) {
        $script:InstalledComponents.NodeJS = $true
        Write-Host "  [" -NoNewline
        Write-Host "INSTALLED" -ForegroundColor $colors.Installed -NoNewline
        Write-Host "] Node.js: $nodeVersion"
    } else {
        Write-Host "  [" -NoNewline
        Write-Host "NOT FOUND" -ForegroundColor $colors.NotInstalled -NoNewline
        Write-Host "] Node.js"
    }

    # Check Python
    $pythonVersion = Get-PythonVersion
    if ($pythonVersion) {
        $script:InstalledComponents.Python = $true
        Write-Host "  [" -NoNewline
        Write-Host "INSTALLED" -ForegroundColor $colors.Installed -NoNewline
        Write-Host "] Python: $pythonVersion"
    } else {
        Write-Host "  [" -NoNewline
        Write-Host "NOT FOUND" -ForegroundColor $colors.NotInstalled -NoNewline
        Write-Host "] Python"
    }

    # Check UV
    $uvVersion = Get-UVVersion
    if ($uvVersion) {
        $script:InstalledComponents.UV = $true
        Write-Host "  [" -NoNewline
        Write-Host "INSTALLED" -ForegroundColor $colors.Installed -NoNewline
        Write-Host "] UV: $uvVersion"
    } else {
        Write-Host "  [" -NoNewline
        Write-Host "NOT FOUND" -ForegroundColor $colors.NotInstalled -NoNewline
        Write-Host "] UV (Python package manager)"
    }

    # Check Claude Code
    $claudeVersion = Get-ClaudeVersion
    if ($claudeVersion) {
        $script:InstalledComponents.ClaudeCode = $true
        Write-Host "  [" -NoNewline
        Write-Host "INSTALLED" -ForegroundColor $colors.Installed -NoNewline
        Write-Host "] Claude Code CLI: $claudeVersion"
    } else {
        Write-Host "  [" -NoNewline
        Write-Host "NOT FOUND" -ForegroundColor $colors.NotInstalled -NoNewline
        Write-Host "] Claude Code CLI"
    }

    Write-Host ""
}

# ============================================================================
# Installation Functions
# ============================================================================

function Install-NodeJS {
    Write-Log "=== Starting Node.js installation ===" "INFO"

    if ($script:InstalledComponents.NodeJS) {
        Write-Success "Node.js is already installed"
        return $true
    }

    Write-Step "Installing Node.js via winget..."
    try {
        Write-Log "Executing: winget install OpenJS.NodeJS.LTS" "DEBUG"
        $result = winget install OpenJS.NodeJS.LTS --accept-source-agreements --accept-package-agreements -e 2>&1
        Write-Log "Winget output: $result" "DEBUG"
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-NodeVersion) {
            $script:InstalledComponents.NodeJS = $true
            Write-Success "Node.js installed successfully"
            return $true
        }
        Write-Log "Node.js not found after winget install" "WARN"
    } catch {
        Write-Log "Winget installation exception: $_" "ERROR"
        Write-Warn "Winget installation failed. Trying alternative method..."
    }

    # Alternative: Download and install manually
    Write-Step "Downloading Node.js installer..."
    $nodeUrl = "https://nodejs.org/dist/v20.11.0/node-v20.11.0-x64.msi"
    $installerPath = "$env:TEMP\node-installer.msi"
    Write-Log "Download URL: $nodeUrl" "DEBUG"
    Write-Log "Installer path: $installerPath" "DEBUG"

    try {
        Invoke-WebRequest -Uri $nodeUrl -OutFile $installerPath -UseBasicParsing
        Write-Log "Download complete" "DEBUG"
        Write-Step "Running Node.js installer..."
        Write-Log "Executing: msiexec.exe /i $installerPath /qn" "DEBUG"
        Start-Process msiexec.exe -ArgumentList "/i `"$installerPath`" /qn" -Wait
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-NodeVersion) {
            $script:InstalledComponents.NodeJS = $true
            Write-Success "Node.js installed successfully"
            return $true
        }
        Write-Log "Node.js not found after MSI install" "ERROR"
    } catch {
        Write-Log "MSI installation exception: $_" "ERROR"
        Write-Err "Failed to install Node.js: $_"
    }

    Write-Err "Node.js installation failed. Please install manually from https://nodejs.org"
    return $false
}

function Install-Python {
    Write-Log "=== Starting Python installation ===" "INFO"

    if ($script:InstalledComponents.Python) {
        Write-Success "Python is already installed"
        return $true
    }

    Write-Step "Installing Python via winget..."
    try {
        Write-Log "Executing: winget install Python.Python.3.12" "DEBUG"
        $result = winget install Python.Python.3.12 --accept-source-agreements --accept-package-agreements -e 2>&1
        Write-Log "Winget output: $result" "DEBUG"
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-PythonVersion) {
            $script:InstalledComponents.Python = $true
            Write-Success "Python installed successfully"
            return $true
        }
        Write-Log "Python not found after winget install" "WARN"
    } catch {
        Write-Log "Winget installation exception: $_" "ERROR"
        Write-Warn "Winget installation failed. Trying alternative method..."
    }

    # Alternative: Download from python.org
    Write-Step "Downloading Python installer..."
    $pythonUrl = "https://www.python.org/ftp/python/3.12.2/python-3.12.2-amd64.exe"
    $installerPath = "$env:TEMP\python-installer.exe"
    Write-Log "Download URL: $pythonUrl" "DEBUG"
    Write-Log "Installer path: $installerPath" "DEBUG"

    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $installerPath -UseBasicParsing
        Write-Log "Download complete" "DEBUG"
        Write-Step "Running Python installer..."
        Write-Log "Executing: $installerPath /quiet InstallAllUsers=0 PrependPath=1" "DEBUG"
        Start-Process $installerPath -ArgumentList "/quiet InstallAllUsers=0 PrependPath=1" -Wait
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-PythonVersion) {
            $script:InstalledComponents.Python = $true
            Write-Success "Python installed successfully"
            return $true
        }
        Write-Log "Python not found after direct install" "ERROR"
    } catch {
        Write-Log "Direct installation exception: $_" "ERROR"
        Write-Err "Failed to install Python: $_"
    }

    Write-Err "Python installation failed. Please install manually from https://python.org"
    return $false
}

function Install-UV {
    Write-Log "=== Starting UV installation ===" "INFO"

    if ($script:InstalledComponents.UV) {
        Write-Success "UV is already installed"
        return $true
    }

    if (-not $script:InstalledComponents.Python) {
        Write-Warn "Python is required to install UV"
        return $false
    }

    Write-Step "Installing UV..."
    try {
        if (Test-CommandExists "py") {
            Write-Log "Executing: py -m pip install uv" "DEBUG"
            $result = py -m pip install uv 2>&1
            Write-Log "pip output: $result" "DEBUG"
        } elseif (Test-CommandExists "python") {
            Write-Log "Executing: python -m pip install uv" "DEBUG"
            $result = python -m pip install uv 2>&1
            Write-Log "pip output: $result" "DEBUG"
        }

        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-UVVersion) {
            $script:InstalledComponents.UV = $true
            Write-Success "UV installed successfully"
            return $true
        }
        Write-Log "UV not found after pip install" "WARN"

        # Try alternative installation via PowerShell
        Write-Step "Trying alternative UV installation..."
        Write-Log "Executing: Invoke-RestMethod https://astral.sh/uv/install.ps1" "DEBUG"
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-UVVersion) {
            $script:InstalledComponents.UV = $true
            Write-Success "UV installed successfully"
            return $true
        }
        Write-Log "UV not found after astral.sh install" "ERROR"
    } catch {
        Write-Log "UV installation exception: $_" "ERROR"
        Write-Err "Failed to install UV: $_"
    }

    Write-Warn "UV installation may have failed. Install manually with: pip install uv"
    return $false
}

function Install-ClaudeCode {
    Write-Log "=== Starting Claude Code CLI installation ===" "INFO"

    if ($script:InstalledComponents.ClaudeCode) {
        Write-Success "Claude Code is already installed"
        return $true
    }

    if (-not $script:InstalledComponents.NodeJS) {
        Write-Warn "Node.js is required to install Claude Code CLI"
        return $false
    }

    Write-Step "Installing Claude Code CLI via npm..."
    try {
        Write-Log "Executing: npm install -g @anthropic-ai/claude-code" "DEBUG"
        $result = npm install -g @anthropic-ai/claude-code 2>&1
        Write-Log "npm output: $result" "DEBUG"
        Refresh-Path
        Start-Sleep -Seconds 2

        if (Get-ClaudeVersion) {
            $script:InstalledComponents.ClaudeCode = $true
            Write-Success "Claude Code CLI installed successfully"
            return $true
        }
        Write-Log "Claude Code CLI not found after npm install" "ERROR"
    } catch {
        Write-Log "npm installation exception: $_" "ERROR"
        Write-Err "Failed to install Claude Code CLI: $_"
    }

    Write-Warn "Claude Code CLI installation may have failed."
    Write-Warn "Install manually with: npm install -g @anthropic-ai/claude-code"
    return $false
}

function Install-PythonPackages {
    param([string[]]$Packages)

    Write-Log "=== Installing Python packages ===" "INFO"
    Write-Log "Packages: $($Packages -join ', ')" "DEBUG"
    Write-Step "Installing Python packages: $($Packages -join ', ')..."

    try {
        foreach ($package in $Packages) {
            Write-Log "Installing package: $package" "DEBUG"
            if (Test-CommandExists "py") {
                Write-Log "Executing: py -m pip install $package --upgrade" "DEBUG"
                $result = py -m pip install $package --upgrade 2>&1
                Write-Log "pip output for $package : $result" "DEBUG"
            } elseif (Test-CommandExists "python") {
                Write-Log "Executing: python -m pip install $package --upgrade" "DEBUG"
                $result = python -m pip install $package --upgrade 2>&1
                Write-Log "pip output for $package : $result" "DEBUG"
            }
        }
        Write-Success "Python packages installed successfully"
        return $true
    } catch {
        Write-Log "Python package installation exception: $_" "ERROR"
        Write-Warn "Some Python packages may have failed to install: $_"
        return $false
    }
}

function Install-WatchdogScript {
    Write-Log "=== Installing Watchdog script ===" "INFO"

    # Create directory for MCP scripts
    $mcpScriptsDir = "$env:USERPROFILE\.claude-mcp-scripts"
    Write-Log "MCP scripts directory: $mcpScriptsDir" "DEBUG"

    if (-not (Test-Path $mcpScriptsDir)) {
        Write-Log "Creating directory: $mcpScriptsDir" "DEBUG"
        New-Item -ItemType Directory -Path $mcpScriptsDir -Force | Out-Null
    }

    $watchdogPath = "$mcpScriptsDir\conversation_watchdog_mcp.py"
    Write-Log "Watchdog script path: $watchdogPath" "DEBUG"

    Write-Step "Installing Conversation Watchdog script..."
    try {
        $WatchdogScript | Out-File -FilePath $watchdogPath -Encoding UTF8 -Force
        Write-Log "Watchdog script written successfully" "INFO"
        Write-Success "Watchdog script installed to: $watchdogPath"
    } catch {
        Write-Log "Failed to write watchdog script: $_" "ERROR"
        Write-Err "Failed to install watchdog script: $_"
    }

    return $watchdogPath
}

# ============================================================================
# MCP Configuration Functions
# ============================================================================

function Show-MCPMenu {
    Write-Header "Select MCP Servers to Install"

    Write-Host "Available MCP Servers:" -ForegroundColor $colors.Menu
    Write-Host ""

    $index = 1
    $menuItems = @{}

    foreach ($key in $MCPServers.Keys | Sort-Object) {
        $server = $MCPServers[$key]
        $menuItems[$index] = $key

        $prereqStatus = ""
        $canInstall = $true

        if ($server.Prerequisites) {
            $missing = @()
            foreach ($prereq in $server.Prerequisites) {
                switch ($prereq) {
                    "nodejs" { if (-not $script:InstalledComponents.NodeJS) { $missing += "Node.js" } }
                    "python" { if (-not $script:InstalledComponents.Python) { $missing += "Python" } }
                    "uv" { if (-not $script:InstalledComponents.UV) { $missing += "UV" } }
                }
            }

            if ($missing.Count -gt 0) {
                $prereqStatus = " [Missing: $($missing -join ', ')]"
                $canInstall = $false
            }
        }

        if ($canInstall) {
            Write-Host "  [$index] " -NoNewline -ForegroundColor $colors.Success
            Write-Host "$($server.Name)" -ForegroundColor $colors.Info
        } else {
            Write-Host "  [$index] " -NoNewline -ForegroundColor $colors.Warning
            Write-Host "$($server.Name)" -NoNewline -ForegroundColor Gray
            Write-Host "$prereqStatus" -ForegroundColor $colors.Warning
        }
        Write-Host "      $($server.Description)" -ForegroundColor Gray
        Write-Host ""
        $index++
    }

    Write-Host "  [A] Select ALL available" -ForegroundColor $colors.Success
    Write-Host "  [Q] Quit without installing MCPs" -ForegroundColor $colors.Warning
    Write-Host ""

    return $menuItems
}

function Get-UserSelections {
    param($MenuItems)

    $selections = @()

    Write-Host "Enter your selections (comma-separated numbers, 'A' for all, or 'Q' to quit):" -ForegroundColor $colors.Menu
    $userInput = Read-Host "Selection"

    if ($userInput -eq 'Q' -or $userInput -eq 'q') {
        return $null
    }

    if ($userInput -eq 'A' -or $userInput -eq 'a') {
        # Return only servers that have their prerequisites met
        $available = @()
        foreach ($key in $MenuItems.Values) {
            $server = $MCPServers[$key]
            $canInstall = $true

            if ($server.Prerequisites) {
                foreach ($prereq in $server.Prerequisites) {
                    switch ($prereq) {
                        "nodejs" { if (-not $script:InstalledComponents.NodeJS) { $canInstall = $false } }
                        "python" { if (-not $script:InstalledComponents.Python) { $canInstall = $false } }
                        "uv" { if (-not $script:InstalledComponents.UV) { $canInstall = $false } }
                    }
                }
            }

            if ($canInstall) {
                $available += $key
            }
        }
        return $available
    }

    $numbers = $userInput -split ',' | ForEach-Object { $_.Trim() }
    foreach ($num in $numbers) {
        try {
            $intNum = [int]$num
            if ($MenuItems.ContainsKey($intNum)) {
                $selections += $MenuItems[$intNum]
            }
        } catch { }
    }

    return $selections
}

function Get-UsernamePathForMCP {
    param(
        [string]$ServerKey,
        [hashtable]$ServerConfig
    )

    Write-Host ""
    Write-Host "Configuration needed for: $($ServerConfig.Name)" -ForegroundColor $colors.Menu
    Write-Host $ServerConfig.UsernamePrompt -ForegroundColor $colors.Info

    # Get default username
    $defaultUsername = $ExecutionContext.InvokeCommand.ExpandString($ServerConfig.UsernameDefault)

    Write-Host "Default username: $defaultUsername" -ForegroundColor Gray
    $username = Read-Host "Username (press Enter for default)"

    if ([string]::IsNullOrWhiteSpace($username)) {
        $username = $defaultUsername
    }

    # Build the path from template
    $basePath = $ServerConfig.PathTemplate -replace '\{USERNAME\}', $username

    # Try to auto-detect the full path with version pattern
    if ($ServerConfig.PathPattern) {
        $found = Get-ChildItem -Path $basePath -Filter $ServerConfig.PathPattern -Directory -ErrorAction SilentlyContinue |
                 Sort-Object Name -Descending |
                 Select-Object -First 1

        if ($found) {
            Write-Host "Auto-detected: $($found.FullName)" -ForegroundColor $colors.Success
            return $found.FullName
        }
    }

    Write-Host "Base path: $basePath" -ForegroundColor Gray
    return $basePath
}

function Get-PathForMCP {
    param(
        [string]$ServerKey,
        [hashtable]$ServerConfig
    )

    Write-Host ""
    Write-Host "Configuration needed for: $($ServerConfig.Name)" -ForegroundColor $colors.Menu
    Write-Host $ServerConfig.PathPrompt -ForegroundColor $colors.Info

    $defaultPath = $ExecutionContext.InvokeCommand.ExpandString($ServerConfig.PathDefault)

    Write-Host "Default: $defaultPath" -ForegroundColor Gray
    $userPath = Read-Host "Path (press Enter for default)"

    if ([string]::IsNullOrWhiteSpace($userPath)) {
        return $defaultPath
    }

    return $userPath
}

function Get-ApiKeyForMCP {
    param(
        [string]$ServerKey,
        [hashtable]$ServerConfig
    )

    Write-Host ""
    Write-Host "API Key needed for: $($ServerConfig.Name)" -ForegroundColor $colors.Menu
    Write-Host $ServerConfig.ApiKeyPrompt -ForegroundColor $colors.Info

    # Show help text if available
    if ($ServerConfig.ApiKeyHelp) {
        Write-Host $ServerConfig.ApiKeyHelp -ForegroundColor $colors.Info
    }

    $apiKey = Read-Host "API Key"
    return $apiKey
}

function Build-MCPConfig {
    param(
        [string[]]$SelectedServers
    )

    Write-Log "=== Building MCP configuration ===" "INFO"
    Write-Log "Selected servers: $($SelectedServers -join ', ')" "DEBUG"

    $mcpConfig = @{
        mcpServers = @{}
    }

    foreach ($serverKey in $SelectedServers) {
        Write-Log "Processing server: $serverKey" "DEBUG"
        $server = $MCPServers[$serverKey]

        if ($null -eq $server.Config) {
            Write-Warn "Skipping $($server.Name) - requires manual configuration"
            Write-Log "Skipping $serverKey - no config defined" "WARN"
            continue
        }

        # Show pre-install note if available
        if ($server.PreInstallNote) {
            Write-Host ""
            Write-Host "Setup note for: $($server.Name)" -ForegroundColor $colors.Menu
            Write-Host $server.PreInstallNote -ForegroundColor $colors.Info
        }

        $config = @{}

        # Handle embedded watchdog script
        if ($server.Type -eq "embedded-python") {
            # Capture only the return value, suppress any other output
            $watchdogPath = (Install-WatchdogScript | Select-Object -Last 1)

            # Install required Python packages
            if ($server.PythonPackages) {
                $null = Install-PythonPackages -Packages $server.PythonPackages
            }

            $config.command = $server.Config.command
            $config.args = @($watchdogPath)
        }
        # Handle username-based path (e.g., PowerBI MCP)
        elseif ($server.RequiresUsername) {
            $path = Get-UsernamePathForMCP -ServerKey $serverKey -ServerConfig $server

            # Validate path exists
            if (-not (Test-Path $path)) {
                Write-Warn "Path does not exist: $path"
                $continue = Read-Host "Continue anyway? (y/n)"
                if ($continue -ne 'y') {
                    Write-Warn "Skipping $($server.Name)"
                    continue
                }
            }

            $config.command = $server.Config.command -replace '\{PATH\}', $path
            # Wrap in @() to ensure single-element arrays remain arrays
            $config.args = @($server.Config.args | ForEach-Object { $_ -replace '\{PATH\}', $path })
        }
        # Handle path substitution
        elseif ($server.RequiresPath) {
            $path = Get-PathForMCP -ServerKey $serverKey -ServerConfig $server

            # Validate path exists
            if (-not (Test-Path $path)) {
                Write-Warn "Path does not exist: $path"
                $continue = Read-Host "Continue anyway? (y/n)"
                if ($continue -ne 'y') {
                    Write-Warn "Skipping $($server.Name)"
                    continue
                }
            }

            $config.command = $server.Config.command -replace '\{PATH\}', $path
            # Wrap in @() to ensure single-element arrays remain arrays
            $config.args = @($server.Config.args | ForEach-Object { $_ -replace '\{PATH\}', $path })
        } else {
            $config.command = $server.Config.command
            # Ensure args remains an array for JSON serialization
            $config.args = @($server.Config.args)
        }

        # Handle API key substitution
        if ($server.RequiresApiKey) {
            $apiKey = Get-ApiKeyForMCP -ServerKey $serverKey -ServerConfig $server

            if ([string]::IsNullOrWhiteSpace($apiKey)) {
                Write-Warn "No API key provided. Skipping $($server.Name)"
                continue
            }

            if ($server.Config.env) {
                $config.env = @{}
                foreach ($envKey in $server.Config.env.Keys) {
                    $config.env[$envKey] = $server.Config.env[$envKey] -replace '\{API_KEY\}', $apiKey
                }
            }
        } elseif ($server.Config.env) {
            $config.env = $server.Config.env
        }

        $mcpConfig.mcpServers[$serverKey] = $config
        Write-Log "Server $serverKey configured: command=$($config.command), args=$($config.args -join ' ')" "DEBUG"
        Write-Success "Configured: $($server.Name)"
    }

    Write-Log "MCP configuration complete. Servers configured: $($mcpConfig.mcpServers.Count)" "INFO"
    return $mcpConfig
}

function Save-MCPConfig {
    param(
        [hashtable]$Config
    )

    $configPath = "$env:APPDATA\Claude\claude_desktop_config.json"
    $configDir = Split-Path $configPath -Parent

    # Ensure directory exists
    if (-not (Test-Path $configDir)) {
        New-Item -ItemType Directory -Path $configDir -Force | Out-Null
    }

    # Backup existing config if present
    if (Test-Path $configPath) {
        $backupPath = "$configPath.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Copy-Item $configPath $backupPath
        Write-Success "Backed up existing config to: $backupPath"

        # Merge with existing config (preserve existing MCPs, skip duplicates)
        try {
            $existingConfig = Get-Content $configPath -Raw | ConvertFrom-Json -AsHashtable

            if ($existingConfig.mcpServers) {
                $skippedMcps = @()
                $addedMcps = @()

                foreach ($key in $Config.mcpServers.Keys) {
                    if ($existingConfig.mcpServers.ContainsKey($key)) {
                        # MCP already exists, skip it (don't overwrite)
                        $skippedMcps += $key
                    } else {
                        # New MCP, add it to existing config
                        $existingConfig.mcpServers[$key] = $Config.mcpServers[$key]
                        $addedMcps += $key
                    }
                }

                # Report skipped duplicates
                if ($skippedMcps.Count -gt 0) {
                    Write-Warn "Skipped existing MCPs (already configured): $($skippedMcps -join ', ')"
                }

                # Report added MCPs
                if ($addedMcps.Count -gt 0) {
                    Write-Success "Added new MCPs: $($addedMcps -join ', ')"
                }

                $Config = $existingConfig
            }
        } catch {
            Write-Warn "Could not parse existing config, creating new one"
        }
    }

    # Add preferences if not present
    if (-not $Config.preferences) {
        $Config.preferences = @{
            chromeExtensionEnabled = $true
        }
    }

    # Save config (use UTF-8 without BOM to avoid JSON parsing errors)
    $jsonConfig = $Config | ConvertTo-Json -Depth 10
    [System.IO.File]::WriteAllText($configPath, $jsonConfig, [System.Text.UTF8Encoding]::new($false))

    Write-Success "Configuration saved to: $configPath"
    return $configPath
}

# ============================================================================
# Windows Forms GUI
# ============================================================================

function Show-GUI {
    Add-Type -AssemblyName System.Windows.Forms
    Add-Type -AssemblyName System.Drawing

    # Initialize logging
    Initialize-Log
    Write-Log "GUI mode started" "INFO"

    # Define colors for the GUI
    # Modern dark theme color palette
    $primaryColor = [System.Drawing.Color]::FromArgb(28, 28, 32)
    $secondaryColor = [System.Drawing.Color]::FromArgb(38, 38, 42)
    $cardColor = [System.Drawing.Color]::FromArgb(48, 48, 54)
    $accentColor = [System.Drawing.Color]::FromArgb(218, 119, 86)  # Claude orange
    $accentHover = [System.Drawing.Color]::FromArgb(238, 139, 106)
    $successColor = [System.Drawing.Color]::FromArgb(78, 201, 140)
    $warningColor = [System.Drawing.Color]::FromArgb(255, 193, 77)
    $errorColor = [System.Drawing.Color]::FromArgb(255, 99, 99)
    $textColor = [System.Drawing.Color]::FromArgb(245, 245, 247)
    $textMuted = [System.Drawing.Color]::FromArgb(150, 150, 158)
    $borderColor = [System.Drawing.Color]::FromArgb(68, 68, 76)

    # Create main form
    $form = New-Object System.Windows.Forms.Form
    $form.Text = "Claude Code & MCP Servers Installer"
    $form.Size = New-Object System.Drawing.Size(800, 700)
    $form.StartPosition = "CenterScreen"
    $form.BackColor = $primaryColor
    $form.ForeColor = $textColor
    $form.FormBorderStyle = "FixedSingle"
    $form.MaximizeBox = $false
    $form.Font = New-Object System.Drawing.Font("Segoe UI", 9)

    # Header Panel with gradient effect
    $headerPanel = New-Object System.Windows.Forms.Panel
    $headerPanel.Size = New-Object System.Drawing.Size(800, 90)
    $headerPanel.Location = New-Object System.Drawing.Point(0, 0)
    $headerPanel.BackColor = $secondaryColor
    $form.Controls.Add($headerPanel)

    # Title - "Claude" in accent color
    $titleClaude = New-Object System.Windows.Forms.Label
    $titleClaude.Text = "Claude"
    $titleClaude.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 22)
    $titleClaude.ForeColor = $accentColor
    $titleClaude.AutoSize = $true
    $titleClaude.Location = New-Object System.Drawing.Point(25, 18)
    $headerPanel.Controls.Add($titleClaude)

    # Title - "MCP Server Installer" in white
    $titleRest = New-Object System.Windows.Forms.Label
    $titleRest.Text = "MCP Server Installer"
    $titleRest.Font = New-Object System.Drawing.Font("Segoe UI Light", 22)
    $titleRest.ForeColor = $textColor
    $titleRest.AutoSize = $true
    $titleRest.Location = New-Object System.Drawing.Point(130, 18)
    $headerPanel.Controls.Add($titleRest)

    # Subtitle Label
    $subtitleLabel = New-Object System.Windows.Forms.Label
    $subtitleLabel.Text = "Configure MCP servers for Claude Desktop"
    $subtitleLabel.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $subtitleLabel.ForeColor = $textMuted
    $subtitleLabel.AutoSize = $true
    $subtitleLabel.Location = New-Object System.Drawing.Point(27, 58)
    $headerPanel.Controls.Add($subtitleLabel)

    # Version Label with badge style
    $versionLabel = New-Object System.Windows.Forms.Label
    $versionLabel.Text = "v2.1.0"
    $versionLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $versionLabel.ForeColor = $accentColor
    $versionLabel.AutoSize = $true
    $versionLabel.Location = New-Object System.Drawing.Point(720, 35)
    $headerPanel.Controls.Add($versionLabel)

    # ========================================
    # Component Status Panel - Card Style
    # ========================================
    $statusPanel = New-Object System.Windows.Forms.Panel
    $statusPanel.Size = New-Object System.Drawing.Size(750, 85)
    $statusPanel.Location = New-Object System.Drawing.Point(20, 100)
    $statusPanel.BackColor = $primaryColor
    $form.Controls.Add($statusPanel)

    # Component status cards
    $componentLabels = @{}
    $componentIcons = @{}
    $componentCards = @{}
    $components = @("Node.js", "Python", "UV", "Claude Code")
    $xPos = 0
    $cardWidth = 175

    foreach ($comp in $components) {
        # Card panel for each component
        $card = New-Object System.Windows.Forms.Panel
        $card.Size = New-Object System.Drawing.Size($cardWidth, 75)
        $card.Location = New-Object System.Drawing.Point($xPos, 5)
        $card.BackColor = $cardColor
        $statusPanel.Controls.Add($card)
        $componentCards[$comp] = $card

        # Status icon (circle)
        $icon = New-Object System.Windows.Forms.Label
        $icon.Text = [char]0x25CF  # Circle bullet
        $icon.Font = New-Object System.Drawing.Font("Segoe UI", 12)
        $icon.ForeColor = $textMuted
        $icon.AutoSize = $true
        $icon.Location = New-Object System.Drawing.Point(12, 12)
        $card.Controls.Add($icon)
        $componentIcons[$comp] = $icon

        # Component name
        $label = New-Object System.Windows.Forms.Label
        $label.Text = $comp
        $label.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 10)
        $label.ForeColor = $textColor
        $label.AutoSize = $true
        $label.Location = New-Object System.Drawing.Point(32, 10)
        $card.Controls.Add($label)

        # Version/status label
        $statusLabel = New-Object System.Windows.Forms.Label
        $statusLabel.Text = "Checking..."
        $statusLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
        $statusLabel.ForeColor = $textMuted
        $statusLabel.AutoSize = $true
        $statusLabel.Location = New-Object System.Drawing.Point(12, 42)
        $card.Controls.Add($statusLabel)
        $componentLabels[$comp] = $statusLabel

        $xPos += $cardWidth + 12
    }

    # ========================================
    # MCP Servers Selection Panel
    # ========================================
    # Section title
    $serversTitle = New-Object System.Windows.Forms.Label
    $serversTitle.Text = "Select MCP Servers"
    $serversTitle.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 11)
    $serversTitle.ForeColor = $textColor
    $serversTitle.AutoSize = $true
    $serversTitle.Location = New-Object System.Drawing.Point(25, 195)
    $form.Controls.Add($serversTitle)

    # Create scrollable panel for server cards
    $scrollPanel = New-Object System.Windows.Forms.Panel
    $scrollPanel.Size = New-Object System.Drawing.Size(750, 265)
    $scrollPanel.Location = New-Object System.Drawing.Point(20, 220)
    $scrollPanel.AutoScroll = $true
    $scrollPanel.BackColor = $primaryColor
    $form.Controls.Add($scrollPanel)

    # Create card-style items for each MCP server
    $serverCheckboxes = @{}
    $yPos = 0

    foreach ($key in $MCPServers.Keys | Sort-Object) {
        $server = $MCPServers[$key]

        # Server item panel (card style)
        $serverCard = New-Object System.Windows.Forms.Panel
        $serverCard.Size = New-Object System.Drawing.Size(710, 65)
        $serverCard.Location = New-Object System.Drawing.Point(5, $yPos)
        $serverCard.BackColor = $cardColor
        $scrollPanel.Controls.Add($serverCard)

        # Checkbox
        $checkbox = New-Object System.Windows.Forms.CheckBox
        $checkbox.Text = ""
        $checkbox.Size = New-Object System.Drawing.Size(20, 20)
        $checkbox.Location = New-Object System.Drawing.Point(15, 22)
        $checkbox.Tag = $key
        $checkbox.Checked = $false
        $checkbox.BackColor = $cardColor
        $serverCard.Controls.Add($checkbox)
        $serverCheckboxes[$key] = $checkbox

        # Server name
        $nameLabel = New-Object System.Windows.Forms.Label
        $nameLabel.Text = $server.Name
        $nameLabel.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 10)
        $nameLabel.ForeColor = $textColor
        $nameLabel.AutoSize = $true
        $nameLabel.Location = New-Object System.Drawing.Point(45, 12)
        $serverCard.Controls.Add($nameLabel)

        # Description
        $descLabel = New-Object System.Windows.Forms.Label
        $descLabel.Text = $server.Description
        $descLabel.Font = New-Object System.Drawing.Font("Segoe UI", 9)
        $descLabel.ForeColor = $textMuted
        $descLabel.AutoSize = $true
        $descLabel.Location = New-Object System.Drawing.Point(45, 34)
        $serverCard.Controls.Add($descLabel)

        # Prerequisites badge (right side)
        if ($server.Prerequisites) {
            $prereqLabel = New-Object System.Windows.Forms.Label
            $prereqLabel.Text = $server.Prerequisites -join ", "
            $prereqLabel.Font = New-Object System.Drawing.Font("Segoe UI", 8)
            $prereqLabel.ForeColor = $warningColor
            $prereqLabel.AutoSize = $true
            $prereqLabel.Location = New-Object System.Drawing.Point(580, 22)
            $prereqLabel.Tag = "prereq_$key"
            $serverCard.Controls.Add($prereqLabel)
        }

        $yPos += 72
    }

    # Select All / Deselect All buttons (smaller, positioned inline)
    $selectAllBtn = New-Object System.Windows.Forms.Button
    $selectAllBtn.Text = "Select All"
    $selectAllBtn.Size = New-Object System.Drawing.Size(90, 28)
    $selectAllBtn.Location = New-Object System.Drawing.Point(25, 490)
    $selectAllBtn.BackColor = $cardColor
    $selectAllBtn.ForeColor = $textColor
    $selectAllBtn.FlatStyle = "Flat"
    $selectAllBtn.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $selectAllBtn.FlatAppearance.BorderColor = $borderColor
    $selectAllBtn.FlatAppearance.BorderSize = 1
    $selectAllBtn.Cursor = [System.Windows.Forms.Cursors]::Hand
    $selectAllBtn.Add_Click({
        foreach ($cb in $serverCheckboxes.Values) {
            if ($cb.Enabled) { $cb.Checked = $true }
        }
    })
    $form.Controls.Add($selectAllBtn)

    $deselectAllBtn = New-Object System.Windows.Forms.Button
    $deselectAllBtn.Text = "Deselect All"
    $deselectAllBtn.Size = New-Object System.Drawing.Size(90, 28)
    $deselectAllBtn.Location = New-Object System.Drawing.Point(125, 490)
    $deselectAllBtn.BackColor = $cardColor
    $deselectAllBtn.ForeColor = $textMuted
    $deselectAllBtn.FlatStyle = "Flat"
    $deselectAllBtn.Font = New-Object System.Drawing.Font("Segoe UI", 9)
    $deselectAllBtn.FlatAppearance.BorderColor = $borderColor
    $deselectAllBtn.FlatAppearance.BorderSize = 1
    $deselectAllBtn.Cursor = [System.Windows.Forms.Cursors]::Hand
    $deselectAllBtn.Add_Click({
        foreach ($cb in $serverCheckboxes.Values) {
            $cb.Checked = $false
        }
    })
    $form.Controls.Add($deselectAllBtn)

    # ========================================
    # Progress Section
    # ========================================
    $progressPanel = New-Object System.Windows.Forms.Panel
    $progressPanel.Size = New-Object System.Drawing.Size(750, 100)
    $progressPanel.Location = New-Object System.Drawing.Point(20, 528)
    $progressPanel.BackColor = $cardColor
    $form.Controls.Add($progressPanel)

    # Status text
    $statusText = New-Object System.Windows.Forms.Label
    $statusText.Text = "Ready to install"
    $statusText.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $statusText.ForeColor = $textColor
    $statusText.Size = New-Object System.Drawing.Size(720, 22)
    $statusText.Location = New-Object System.Drawing.Point(15, 12)
    $progressPanel.Controls.Add($statusText)

    # Progress bar (styled)
    $progressBar = New-Object System.Windows.Forms.ProgressBar
    $progressBar.Size = New-Object System.Drawing.Size(720, 8)
    $progressBar.Location = New-Object System.Drawing.Point(15, 40)
    $progressBar.Style = "Continuous"
    $progressPanel.Controls.Add($progressBar)

    # Log textbox
    $logTextBox = New-Object System.Windows.Forms.TextBox
    $logTextBox.Multiline = $true
    $logTextBox.ScrollBars = "Vertical"
    $logTextBox.ReadOnly = $true
    $logTextBox.BackColor = $secondaryColor
    $logTextBox.ForeColor = $textMuted
    $logTextBox.Font = New-Object System.Drawing.Font("Cascadia Code", 8)
    $logTextBox.Size = New-Object System.Drawing.Size(720, 40)
    $logTextBox.Location = New-Object System.Drawing.Point(15, 55)
    $logTextBox.BorderStyle = "None"
    $progressPanel.Controls.Add($logTextBox)

    # ========================================
    # Action Buttons
    # ========================================
    $installBtn = New-Object System.Windows.Forms.Button
    $installBtn.Text = "Install Selected"
    $installBtn.Size = New-Object System.Drawing.Size(150, 42)
    $installBtn.Location = New-Object System.Drawing.Point(500, 640)
    $installBtn.BackColor = $accentColor
    $installBtn.ForeColor = $textColor
    $installBtn.Font = New-Object System.Drawing.Font("Segoe UI Semibold", 10)
    $installBtn.FlatStyle = "Flat"
    $installBtn.FlatAppearance.BorderSize = 0
    $installBtn.Cursor = [System.Windows.Forms.Cursors]::Hand
    $form.Controls.Add($installBtn)

    $cancelBtn = New-Object System.Windows.Forms.Button
    $cancelBtn.Text = "Cancel"
    $cancelBtn.Size = New-Object System.Drawing.Size(100, 42)
    $cancelBtn.Location = New-Object System.Drawing.Point(660, 640)
    $cancelBtn.BackColor = $cardColor
    $cancelBtn.ForeColor = $textMuted
    $cancelBtn.Font = New-Object System.Drawing.Font("Segoe UI", 10)
    $cancelBtn.FlatStyle = "Flat"
    $cancelBtn.FlatAppearance.BorderColor = $borderColor
    $cancelBtn.FlatAppearance.BorderSize = 1
    $cancelBtn.Cursor = [System.Windows.Forms.Cursors]::Hand
    $cancelBtn.Add_Click({ $form.Close() })
    $form.Controls.Add($cancelBtn)

    # ========================================
    # Helper function to update log
    # ========================================
    $script:GuiLogMessages = @()

    function Update-GuiLog {
        param([string]$Message, [string]$Level = "INFO")
        $timestamp = Get-Date -Format "HH:mm:ss"
        $logEntry = "[$timestamp] $Message"
        $script:GuiLogMessages += $logEntry
        $logTextBox.Text = ($script:GuiLogMessages | Select-Object -Last 10) -join "`r`n"
        $logTextBox.SelectionStart = $logTextBox.Text.Length
        $logTextBox.ScrollToCaret()
        $statusText.Text = $Message
        Write-Log $Message $Level
        [System.Windows.Forms.Application]::DoEvents()
    }

    # ========================================
    # Check components on load
    # ========================================
    $form.Add_Shown({
        Update-GuiLog "Checking system components..."

        # Check Node.js
        $nodeVersion = Get-NodeVersion
        if ($nodeVersion) {
            $script:InstalledComponents.NodeJS = $true
            $componentIcons["Node.js"].ForeColor = $successColor
            $componentLabels["Node.js"].Text = $nodeVersion
            $componentLabels["Node.js"].ForeColor = $successColor
        } else {
            $componentIcons["Node.js"].ForeColor = $errorColor
            $componentLabels["Node.js"].Text = "Not installed"
            $componentLabels["Node.js"].ForeColor = $errorColor
        }
        [System.Windows.Forms.Application]::DoEvents()

        # Check Python
        $pythonVersion = Get-PythonVersion
        if ($pythonVersion) {
            $script:InstalledComponents.Python = $true
            $componentIcons["Python"].ForeColor = $successColor
            $componentLabels["Python"].Text = $pythonVersion
            $componentLabels["Python"].ForeColor = $successColor
        } else {
            $componentIcons["Python"].ForeColor = $errorColor
            $componentLabels["Python"].Text = "Not installed"
            $componentLabels["Python"].ForeColor = $errorColor
        }
        [System.Windows.Forms.Application]::DoEvents()

        # Check UV
        $uvVersion = Get-UVVersion
        if ($uvVersion) {
            $script:InstalledComponents.UV = $true
            $componentIcons["UV"].ForeColor = $successColor
            $componentLabels["UV"].Text = $uvVersion
            $componentLabels["UV"].ForeColor = $successColor
        } else {
            $componentIcons["UV"].ForeColor = $errorColor
            $componentLabels["UV"].Text = "Not installed"
            $componentLabels["UV"].ForeColor = $errorColor
        }
        [System.Windows.Forms.Application]::DoEvents()

        # Check Claude Code
        $claudeVersion = Get-ClaudeVersion
        if ($claudeVersion) {
            $script:InstalledComponents.ClaudeCode = $true
            $componentIcons["Claude Code"].ForeColor = $successColor
            $componentLabels["Claude Code"].Text = $claudeVersion
            $componentLabels["Claude Code"].ForeColor = $successColor
        } else {
            $componentIcons["Claude Code"].ForeColor = $warningColor
            $componentLabels["Claude Code"].Text = "Not installed"
            $componentLabels["Claude Code"].ForeColor = $warningColor
        }
        [System.Windows.Forms.Application]::DoEvents()

        # Update checkbox states based on prerequisites
        foreach ($key in $MCPServers.Keys) {
            $server = $MCPServers[$key]
            $canInstall = $true
            $missingPrereqs = @()

            if ($server.Prerequisites) {
                foreach ($prereq in $server.Prerequisites) {
                    switch ($prereq) {
                        "nodejs" {
                            if (-not $script:InstalledComponents.NodeJS) {
                                $canInstall = $false
                                $missingPrereqs += "Node.js"
                            }
                        }
                        "python" {
                            if (-not $script:InstalledComponents.Python) {
                                $canInstall = $false
                                $missingPrereqs += "Python"
                            }
                        }
                        "uv" {
                            if (-not $script:InstalledComponents.UV) {
                                $canInstall = $false
                                $missingPrereqs += "UV"
                            }
                        }
                    }
                }
            }

            if (-not $canInstall) {
                $serverCheckboxes[$key].Enabled = $false
                $serverCheckboxes[$key].ForeColor = $textMuted
            }
        }

        Update-GuiLog "Component check complete. Ready to install."
    })

    # ========================================
    # Install button click handler
    # ========================================
    $installBtn.Add_Click({
        # Get selected servers
        $selectedServers = @()
        foreach ($key in $serverCheckboxes.Keys) {
            if ($serverCheckboxes[$key].Checked) {
                $selectedServers += $key
            }
        }

        if ($selectedServers.Count -eq 0) {
            [System.Windows.Forms.MessageBox]::Show(
                "Please select at least one MCP server to install.",
                "No Selection",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            )
            return
        }

        # Disable controls during installation
        $installBtn.Enabled = $false
        $selectAllBtn.Enabled = $false
        $deselectAllBtn.Enabled = $false
        foreach ($cb in $serverCheckboxes.Values) {
            $cb.Enabled = $false
        }

        $progressBar.Value = 0
        $progressBar.Maximum = $selectedServers.Count * 2  # 2 steps per server

        Update-GuiLog "Starting installation of $($selectedServers.Count) server(s)..."

        $configuredServers = @{
            mcpServers = @{}
        }
        $currentStep = 0

        foreach ($serverKey in $selectedServers) {
            $server = $MCPServers[$serverKey]
            Update-GuiLog "Configuring: $($server.Name)..."
            $currentStep++
            $progressBar.Value = $currentStep

            try {
                $config = @{}

                # Handle embedded watchdog script
                if ($server.Type -eq "embedded-python") {
                    Update-GuiLog "Installing watchdog script..."
                    # Capture only the return value, suppress any other output
                    $watchdogPath = (Install-WatchdogScript | Select-Object -Last 1)

                    if ($server.PythonPackages) {
                        Update-GuiLog "Installing Python packages..."
                        $null = Install-PythonPackages -Packages $server.PythonPackages
                    }

                    $config.command = $server.Config.command
                    $config.args = @($watchdogPath)
                }
                # Handle username-based path (e.g., PowerBI MCP)
                elseif ($server.RequiresUsername) {
                    # Show username input dialog
                    $userForm = New-Object System.Windows.Forms.Form
                    $userForm.Text = "Configure $($server.Name)"
                    $userForm.Size = New-Object System.Drawing.Size(500, 180)
                    $userForm.StartPosition = "CenterParent"
                    $userForm.BackColor = $primaryColor
                    $userForm.ForeColor = $textColor
                    $userForm.FormBorderStyle = "FixedDialog"
                    $userForm.MaximizeBox = $false
                    $userForm.MinimizeBox = $false

                    $userLabel = New-Object System.Windows.Forms.Label
                    $userLabel.Text = $server.UsernamePrompt
                    $userLabel.Location = New-Object System.Drawing.Point(20, 20)
                    $userLabel.Size = New-Object System.Drawing.Size(450, 40)
                    $userForm.Controls.Add($userLabel)

                    $userTextBox = New-Object System.Windows.Forms.TextBox
                    $userTextBox.Size = New-Object System.Drawing.Size(440, 25)
                    $userTextBox.Location = New-Object System.Drawing.Point(20, 65)
                    $userTextBox.BackColor = $secondaryColor
                    $userTextBox.ForeColor = $textColor
                    $userTextBox.Text = $ExecutionContext.InvokeCommand.ExpandString($server.UsernameDefault)
                    $userForm.Controls.Add($userTextBox)

                    $userOkBtn = New-Object System.Windows.Forms.Button
                    $userOkBtn.Text = "OK"
                    $userOkBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $userOkBtn.Location = New-Object System.Drawing.Point(290, 105)
                    $userOkBtn.BackColor = $successColor
                    $userOkBtn.ForeColor = $secondaryColor
                    $userOkBtn.FlatStyle = "Flat"
                    $userOkBtn.DialogResult = [System.Windows.Forms.DialogResult]::OK
                    $userForm.Controls.Add($userOkBtn)
                    $userForm.AcceptButton = $userOkBtn

                    $userSkipBtn = New-Object System.Windows.Forms.Button
                    $userSkipBtn.Text = "Skip"
                    $userSkipBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $userSkipBtn.Location = New-Object System.Drawing.Point(380, 105)
                    $userSkipBtn.BackColor = [System.Drawing.Color]::FromArgb(60, 60, 60)
                    $userSkipBtn.ForeColor = $textColor
                    $userSkipBtn.FlatStyle = "Flat"
                    $userSkipBtn.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
                    $userForm.Controls.Add($userSkipBtn)

                    $userResult = $userForm.ShowDialog()
                    if ($userResult -eq [System.Windows.Forms.DialogResult]::OK) {
                        $username = $userTextBox.Text
                        $basePath = $server.PathTemplate -replace '\{USERNAME\}', $username

                        # Try to auto-detect the full path with version pattern
                        $path = $basePath
                        if ($server.PathPattern) {
                            $found = Get-ChildItem -Path $basePath -Filter $server.PathPattern -Directory -ErrorAction SilentlyContinue |
                                     Sort-Object Name -Descending | Select-Object -First 1
                            if ($found) {
                                $path = $found.FullName
                                Update-GuiLog "Auto-detected: $path"
                            }
                        }

                        $config.command = $server.Config.command -replace '\{PATH\}', $path
                        $config.args = @($server.Config.args | ForEach-Object { $_ -replace '\{PATH\}', $path })
                    } else {
                        Update-GuiLog "Skipped: $($server.Name)"
                        $currentStep++
                        $progressBar.Value = $currentStep
                        continue
                    }
                }
                # Handle path substitution
                elseif ($server.RequiresPath) {
                    # Show path input dialog
                    $pathForm = New-Object System.Windows.Forms.Form
                    $pathForm.Text = "Configure $($server.Name)"
                    $pathForm.Size = New-Object System.Drawing.Size(600, 200)
                    $pathForm.StartPosition = "CenterParent"
                    $pathForm.BackColor = $primaryColor
                    $pathForm.ForeColor = $textColor
                    $pathForm.FormBorderStyle = "FixedDialog"
                    $pathForm.MaximizeBox = $false
                    $pathForm.MinimizeBox = $false

                    $pathLabel = New-Object System.Windows.Forms.Label
                    $pathLabel.Text = $server.PathPrompt
                    $pathLabel.Location = New-Object System.Drawing.Point(20, 20)
                    $pathLabel.Size = New-Object System.Drawing.Size(550, 40)
                    $pathForm.Controls.Add($pathLabel)

                    $pathTextBox = New-Object System.Windows.Forms.TextBox
                    $pathTextBox.Size = New-Object System.Drawing.Size(450, 25)
                    $pathTextBox.Location = New-Object System.Drawing.Point(20, 70)
                    $pathTextBox.BackColor = $secondaryColor
                    $pathTextBox.ForeColor = $textColor

                    # Try auto-detect
                    $defaultPath = $ExecutionContext.InvokeCommand.ExpandString($server.PathDefault)
                    if ($server.PathPattern) {
                        $found = Get-ChildItem -Path $defaultPath -Filter $server.PathPattern -Directory -ErrorAction SilentlyContinue |
                                 Sort-Object Name -Descending | Select-Object -First 1
                        if ($found) {
                            $pathTextBox.Text = $found.FullName
                        } else {
                            $pathTextBox.Text = $defaultPath
                        }
                    } else {
                        $pathTextBox.Text = $defaultPath
                    }
                    $pathForm.Controls.Add($pathTextBox)

                    $browseBtn = New-Object System.Windows.Forms.Button
                    $browseBtn.Text = "Browse..."
                    $browseBtn.Size = New-Object System.Drawing.Size(80, 25)
                    $browseBtn.Location = New-Object System.Drawing.Point(480, 70)
                    $browseBtn.BackColor = $accentColor
                    $browseBtn.ForeColor = $textColor
                    $browseBtn.FlatStyle = "Flat"
                    $browseBtn.Add_Click({
                        $folderBrowser = New-Object System.Windows.Forms.FolderBrowserDialog
                        $folderBrowser.Description = "Select the installation folder"
                        if ($folderBrowser.ShowDialog() -eq "OK") {
                            $pathTextBox.Text = $folderBrowser.SelectedPath
                        }
                    })
                    $pathForm.Controls.Add($browseBtn)

                    $okBtn = New-Object System.Windows.Forms.Button
                    $okBtn.Text = "OK"
                    $okBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $okBtn.Location = New-Object System.Drawing.Point(400, 120)
                    $okBtn.BackColor = $successColor
                    $okBtn.ForeColor = $secondaryColor
                    $okBtn.FlatStyle = "Flat"
                    $okBtn.DialogResult = [System.Windows.Forms.DialogResult]::OK
                    $pathForm.Controls.Add($okBtn)
                    $pathForm.AcceptButton = $okBtn

                    $skipBtn = New-Object System.Windows.Forms.Button
                    $skipBtn.Text = "Skip"
                    $skipBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $skipBtn.Location = New-Object System.Drawing.Point(490, 120)
                    $skipBtn.BackColor = [System.Drawing.Color]::FromArgb(60, 60, 60)
                    $skipBtn.ForeColor = $textColor
                    $skipBtn.FlatStyle = "Flat"
                    $skipBtn.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
                    $pathForm.Controls.Add($skipBtn)

                    $result = $pathForm.ShowDialog()
                    if ($result -eq [System.Windows.Forms.DialogResult]::OK) {
                        $path = $pathTextBox.Text
                        $config.command = $server.Config.command -replace '\{PATH\}', $path
                        $config.args = @($server.Config.args | ForEach-Object { $_ -replace '\{PATH\}', $path })
                    } else {
                        Update-GuiLog "Skipped: $($server.Name)"
                        $currentStep++
                        $progressBar.Value = $currentStep
                        continue
                    }
                } else {
                    $config.command = $server.Config.command
                    $config.args = @($server.Config.args)
                }

                # Handle API key substitution
                if ($server.RequiresApiKey) {
                    $apiForm = New-Object System.Windows.Forms.Form
                    $apiForm.Text = "API Key for $($server.Name)"
                    $apiForm.Size = New-Object System.Drawing.Size(500, 250)
                    $apiForm.StartPosition = "CenterParent"
                    $apiForm.BackColor = $primaryColor
                    $apiForm.ForeColor = $textColor
                    $apiForm.FormBorderStyle = "FixedDialog"
                    $apiForm.MaximizeBox = $false
                    $apiForm.MinimizeBox = $false

                    $apiLabel = New-Object System.Windows.Forms.Label
                    $apiLabel.Text = $server.ApiKeyPrompt
                    $apiLabel.Location = New-Object System.Drawing.Point(20, 20)
                    $apiLabel.AutoSize = $true
                    $apiForm.Controls.Add($apiLabel)

                    if ($server.ApiKeyHelp) {
                        $helpLabel = New-Object System.Windows.Forms.Label
                        $helpLabel.Text = $server.ApiKeyHelp
                        $helpLabel.Location = New-Object System.Drawing.Point(20, 50)
                        $helpLabel.Size = New-Object System.Drawing.Size(450, 80)
                        $helpLabel.ForeColor = $textMuted
                        $apiForm.Controls.Add($helpLabel)
                    }

                    $apiTextBox = New-Object System.Windows.Forms.TextBox
                    $apiTextBox.Size = New-Object System.Drawing.Size(440, 25)
                    $apiTextBox.Location = New-Object System.Drawing.Point(20, 140)
                    $apiTextBox.BackColor = $secondaryColor
                    $apiTextBox.ForeColor = $textColor
                    $apiTextBox.UseSystemPasswordChar = $true
                    $apiForm.Controls.Add($apiTextBox)

                    $apiOkBtn = New-Object System.Windows.Forms.Button
                    $apiOkBtn.Text = "OK"
                    $apiOkBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $apiOkBtn.Location = New-Object System.Drawing.Point(290, 175)
                    $apiOkBtn.BackColor = $successColor
                    $apiOkBtn.ForeColor = $secondaryColor
                    $apiOkBtn.FlatStyle = "Flat"
                    $apiOkBtn.DialogResult = [System.Windows.Forms.DialogResult]::OK
                    $apiForm.Controls.Add($apiOkBtn)
                    $apiForm.AcceptButton = $apiOkBtn

                    $apiSkipBtn = New-Object System.Windows.Forms.Button
                    $apiSkipBtn.Text = "Skip"
                    $apiSkipBtn.Size = New-Object System.Drawing.Size(80, 30)
                    $apiSkipBtn.Location = New-Object System.Drawing.Point(380, 175)
                    $apiSkipBtn.BackColor = [System.Drawing.Color]::FromArgb(60, 60, 60)
                    $apiSkipBtn.ForeColor = $textColor
                    $apiSkipBtn.FlatStyle = "Flat"
                    $apiSkipBtn.DialogResult = [System.Windows.Forms.DialogResult]::Cancel
                    $apiForm.Controls.Add($apiSkipBtn)

                    $apiResult = $apiForm.ShowDialog()
                    if ($apiResult -eq [System.Windows.Forms.DialogResult]::OK -and -not [string]::IsNullOrWhiteSpace($apiTextBox.Text)) {
                        $apiKey = $apiTextBox.Text
                        if ($server.Config.env) {
                            $config.env = @{}
                            foreach ($envKey in $server.Config.env.Keys) {
                                $config.env[$envKey] = $server.Config.env[$envKey] -replace '\{API_KEY\}', $apiKey
                            }
                        }
                    } else {
                        Update-GuiLog "Skipped: $($server.Name) (no API key)"
                        $currentStep++
                        $progressBar.Value = $currentStep
                        continue
                    }
                } elseif ($server.Config.env) {
                    $config.env = $server.Config.env
                }

                $configuredServers.mcpServers[$serverKey] = $config
                Update-GuiLog "Configured: $($server.Name)"

            } catch {
                Update-GuiLog "Error configuring $($server.Name): $_"
                Write-Log "Error: $_" "ERROR"
            }

            $currentStep++
            $progressBar.Value = $currentStep
        }

        # Save configuration
        if ($configuredServers.mcpServers.Count -gt 0) {
            Update-GuiLog "Saving configuration..."
            try {
                $configPath = Save-MCPConfig -Config $configuredServers
                $progressBar.Value = $progressBar.Maximum
                Update-GuiLog "Installation complete!"

                [System.Windows.Forms.MessageBox]::Show(
                    "Installation completed successfully!`n`nConfiguration saved to:`n$configPath`n`nPlease restart Claude Desktop to load the new MCP servers.",
                    "Installation Complete",
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Information
                )
            } catch {
                Update-GuiLog "Error saving configuration: $_"
                [System.Windows.Forms.MessageBox]::Show(
                    "Error saving configuration:`n$_",
                    "Error",
                    [System.Windows.Forms.MessageBoxButtons]::OK,
                    [System.Windows.Forms.MessageBoxIcon]::Error
                )
            }
        } else {
            Update-GuiLog "No servers were configured."
            [System.Windows.Forms.MessageBox]::Show(
                "No MCP servers were configured.",
                "Warning",
                [System.Windows.Forms.MessageBoxButtons]::OK,
                [System.Windows.Forms.MessageBoxIcon]::Warning
            )
        }

        # Re-enable controls
        $installBtn.Enabled = $true
        $selectAllBtn.Enabled = $true
        $deselectAllBtn.Enabled = $true
        foreach ($key in $serverCheckboxes.Keys) {
            $server = $MCPServers[$key]
            $canInstall = $true
            if ($server.Prerequisites) {
                foreach ($prereq in $server.Prerequisites) {
                    switch ($prereq) {
                        "nodejs" { if (-not $script:InstalledComponents.NodeJS) { $canInstall = $false } }
                        "python" { if (-not $script:InstalledComponents.Python) { $canInstall = $false } }
                        "uv" { if (-not $script:InstalledComponents.UV) { $canInstall = $false } }
                    }
                }
            }
            $serverCheckboxes[$key].Enabled = $canInstall
        }
    })

    # Show the form
    [void]$form.ShowDialog()
}

# ============================================================================
# Main Installation Flow
# ============================================================================

function Main {
    Clear-Host

    # Initialize logging
    Initialize-Log

    Write-Header "Claude Code & MCP Servers Installer v2.0"
    Write-Host "This installer will set up Claude Code CLI and configure MCP servers" -ForegroundColor $colors.Info
    Write-Host "for Claude Desktop on this machine." -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "Current User: $env:USERNAME" -ForegroundColor $colors.Info
    Write-Host "User Profile: $env:USERPROFILE" -ForegroundColor $colors.Info
    Write-Host ""
    Write-Host "Log file: $script:LogFile" -ForegroundColor Gray
    Write-Host ""

    Write-Log "User: $env:USERNAME" "INFO"
    Write-Log "User Profile: $env:USERPROFILE" "INFO"

    # Check for admin rights
    $isAdmin = Test-Administrator
    Write-Log "Administrator: $isAdmin" "INFO"
    if (-not $isAdmin) {
        Write-Warn "Running without administrator privileges."
        Write-Warn "Some installations may require elevated permissions."
        Write-Host ""
    }

    # -------------------------------------------------------------------------
    # Step 1: Show current component status
    # -------------------------------------------------------------------------

    Show-ComponentStatus

    Write-Host "Press any key to continue..." -ForegroundColor $colors.Info
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

    # -------------------------------------------------------------------------
    # Step 2: Ask about installing missing prerequisites
    # -------------------------------------------------------------------------

    if (-not $SkipPrerequisites) {
        $missingComponents = @()

        if (-not $script:InstalledComponents.NodeJS) { $missingComponents += "Node.js" }
        if (-not $script:InstalledComponents.Python) { $missingComponents += "Python" }
        if (-not $script:InstalledComponents.UV) { $missingComponents += "UV" }
        if (-not $script:InstalledComponents.ClaudeCode) { $missingComponents += "Claude Code CLI" }

        if ($missingComponents.Count -gt 0) {
            Write-Header "Install Missing Components"

            Write-Host "The following components are not installed:" -ForegroundColor $colors.Warning
            foreach ($comp in $missingComponents) {
                Write-Host "  - $comp" -ForegroundColor $colors.Warning
            }
            Write-Host ""

            $installChoice = Read-Host "Would you like to install missing components? (y/n)"

            if ($installChoice -eq 'y' -or $installChoice -eq 'Y') {
                Write-Host ""

                if (-not $script:InstalledComponents.NodeJS) {
                    Install-NodeJS | Out-Null
                }

                if (-not $script:InstalledComponents.Python) {
                    Install-Python | Out-Null
                }

                if (-not $script:InstalledComponents.UV -and $script:InstalledComponents.Python) {
                    Install-UV | Out-Null
                }

                if (-not $script:InstalledComponents.ClaudeCode -and $script:InstalledComponents.NodeJS) {
                    Install-ClaudeCode | Out-Null
                }

                Write-Host ""
                Write-Host "Component installation complete. Updated status:" -ForegroundColor $colors.Info
                Show-ComponentStatus
            }
        } else {
            Write-Success "All components are already installed!"
            Write-Host ""
        }
    }

    Write-Host "Press any key to continue to MCP server selection..." -ForegroundColor $colors.Info
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

    # -------------------------------------------------------------------------
    # Step 3: MCP Server Selection
    # -------------------------------------------------------------------------

    $menuItems = Show-MCPMenu
    $selectedServers = Get-UserSelections -MenuItems $menuItems

    if ($null -eq $selectedServers -or $selectedServers.Count -eq 0) {
        Write-Warn "No MCP servers selected. Exiting."
        Write-Host ""
        Write-Host "Press any key to exit..." -ForegroundColor $colors.Info
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        return
    }

    Write-Host ""
    Write-Host "Selected servers:" -ForegroundColor $colors.Success
    foreach ($server in $selectedServers) {
        Write-Host "  - $($MCPServers[$server].Name)" -ForegroundColor $colors.Info
    }
    Write-Host ""

    # -------------------------------------------------------------------------
    # Step 4: Build and Save Configuration
    # -------------------------------------------------------------------------

    Write-Header "Configuring MCP Servers"

    $mcpConfig = Build-MCPConfig -SelectedServers $selectedServers

    if ($mcpConfig.mcpServers.Count -gt 0) {
        $configPath = Save-MCPConfig -Config $mcpConfig
        Write-Log "Configuration saved to: $configPath" "INFO"

        Write-Host ""
        Write-Success "Installation Complete!"
        Write-Host ""
        Write-Host "Configuration saved to:" -ForegroundColor $colors.Info
        Write-Host "  $configPath" -ForegroundColor $colors.Success
        Write-Host ""
        Write-Host "Next steps:" -ForegroundColor $colors.Info
        Write-Host "  1. Restart Claude Desktop to load the new MCP servers" -ForegroundColor $colors.Info
        Write-Host "  2. Use 'claude' command in terminal to access Claude Code CLI" -ForegroundColor $colors.Info
        Write-Host ""
        Write-Host "For troubleshooting, check the log file:" -ForegroundColor Gray
        Write-Host "  $script:LogFile" -ForegroundColor Gray
        Write-Host ""
        Write-Log "=== Installation completed successfully ===" "INFO"
    } else {
        Write-Warn "No MCP servers were configured."
        Write-Log "No MCP servers were configured" "WARN"
    }

    Write-Log "=== Installer finished ===" "INFO"
    Write-Host "Press any key to exit..." -ForegroundColor $colors.Info
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}

# Run the installer (unless -NoRun is specified for importing functions only)
if (-not $NoRun) {
    Main
}
