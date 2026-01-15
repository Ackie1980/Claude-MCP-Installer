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

# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging():
    """Configure logging with file and console output."""
    logger = logging.getLogger("conversation_watchdog")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers
    logger.handlers = []

    # File handler - logs everything to file
    file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)

    # Also log to stderr for MCP server output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)

    return logger

# Initialize logger
log = setup_logging()
log.info("=" * 60)
log.info("WATCHDOG MCP SERVER STARTING")
log.info(f"Data directory: {WATCHDOG_DATA_DIR}")
log.info(f"Log file: {LOG_FILE}")
log.info(f"Auto-start enabled: {AUTO_START_ENABLED}")
log.info("=" * 60)


# ============================================================================
# Data Models
# ============================================================================

class TaskStatus(str, Enum):
    """Status of a tracked task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    TRUNCATED = "truncated"
    NEEDS_CONTINUATION = "needs_continuation"


class TruncationType(str, Enum):
    """Type of truncation detected."""
    MID_SENTENCE = "mid_sentence"
    MID_WORD = "mid_word"
    MID_CODE_BLOCK = "mid_code_block"
    MID_LIST = "mid_list"
    INCOMPLETE_THOUGHT = "incomplete_thought"
    CLEAN_END = "clean_end"
    UNKNOWN = "unknown"


@dataclass
class TrackedTask:
    """A task being tracked by the watchdog."""
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


# ============================================================================
# Storage Utilities
# ============================================================================

def load_tasks() -> Dict[str, TrackedTask]:
    """Load tracked tasks from storage."""
    log.debug(f"Loading tasks from {TASKS_FILE}")
    if not TASKS_FILE.exists():
        log.debug("Tasks file does not exist, returning empty dict")
        return {}
    try:
        with open(TASKS_FILE, 'r') as f:
            data = json.load(f)
            tasks = {k: TrackedTask(**v) for k, v in data.items()}
            log.debug(f"Loaded {len(tasks)} tasks from storage")
            return tasks
    except (json.JSONDecodeError, TypeError) as e:
        log.error(f"Error loading tasks: {e}")
        return {}


def save_tasks(tasks: Dict[str, TrackedTask]) -> None:
    """Save tracked tasks to storage."""
    log.debug(f"Saving {len(tasks)} tasks to {TASKS_FILE}")
    with open(TASKS_FILE, 'w') as f:
        json.dump({k: asdict(v) for k, v in tasks.items()}, f, indent=2)
    log.info(f"Saved {len(tasks)} tasks to storage")


def load_history() -> List[Dict[str, Any]]:
    """Load task history."""
    log.debug(f"Loading history from {HISTORY_FILE}")
    if not HISTORY_FILE.exists():
        log.debug("History file does not exist, returning empty list")
        return []
    try:
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            log.debug(f"Loaded {len(history)} history entries")
            return history
    except json.JSONDecodeError as e:
        log.error(f"Error loading history: {e}")
        return []


def save_history(history: List[Dict[str, Any]]) -> None:
    """Save task history."""
    log.debug(f"Saving history ({len(history)} entries)")
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-100:], f, indent=2)  # Keep last 100 entries
    log.info(f"Saved {min(len(history), 100)} history entries")


def generate_task_id(question: str) -> str:
    """Generate a unique task ID based on question content and time."""
    content = f"{question}_{datetime.now().isoformat()}"
    task_id = hashlib.sha256(content.encode()).hexdigest()[:12]
    log.debug(f"Generated task ID: {task_id}")
    return task_id


def get_or_create_auto_task(question: str = None) -> Optional[TrackedTask]:
    """Get the current active task or auto-create one if auto-start is enabled.

    This ensures the watchdog ALWAYS runs - no matter what.
    """
    log.debug("get_or_create_auto_task called")
    tasks = load_tasks()

    # Find any in-progress task
    in_progress = [t for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS]
    if in_progress:
        # Return most recent
        task = max(in_progress, key=lambda t: t.updated_at)
        log.info(f"Found existing in-progress task: {task.task_id}")
        return task

    # Auto-create a new task if enabled (which is ALWAYS)
    if AUTO_START_ENABLED:
        task_id = generate_task_id(question or "Auto-tracked conversation")
        now = datetime.now().isoformat()

        task = TrackedTask(
            task_id=task_id,
            original_question=question or "Auto-tracked conversation - watchdog always active",
            status=TaskStatus.IN_PROGRESS,
            created_at=now,
            updated_at=now,
            checkpoints=[{
                "timestamp": now,
                "description": "Auto-started by watchdog (always-on mode)",
                "completion_percentage": 0,
                "checkpoint_number": 1
            }]
        )

        tasks[task_id] = task
        save_tasks(tasks)
        log.info(f"AUTO-CREATED new task: {task_id} (always-on mode)")
        log.info(f"Question: {(question or 'Auto-tracked conversation')[:100]}...")
        return task

    log.warning("Auto-start disabled and no active tasks found")
    return None


def ensure_watchdog_active() -> TrackedTask:
    """Ensure the watchdog is active. Creates a task if none exists.

    Called automatically - watchdog ALWAYS runs.
    """
    log.debug("ensure_watchdog_active called")
    return get_or_create_auto_task()


# ============================================================================
# Truncation Detection
# ============================================================================

TRUNCATION_PATTERNS = {
    TruncationType.MID_SENTENCE: [
        r'[a-zA-Z0-9]\s*$',  # Ends with letter/number and maybe space
        r',\s*$',  # Ends with comma
        r':\s*$',  # Ends with colon (expecting more)
        r'\band\s*$',  # Ends with "and"
        r'\bor\s*$',  # Ends with "or"
        r'\bthe\s*$',  # Ends with "the"
        r'\ba\s*$',  # Ends with "a"
        r'\bto\s*$',  # Ends with "to"
        r'\bfor\s*$',  # Ends with "for"
        r'\bwith\s*$',  # Ends with "with"
    ],
    TruncationType.MID_WORD: [
        r'[a-zA-Z]{2,}$',  # Ends with partial word (no punctuation)
    ],
    TruncationType.MID_CODE_BLOCK: [
        r'```[a-zA-Z]*\n(?!.*```)',  # Unclosed code block
        r'`[^`]+$',  # Unclosed inline code
        r'\{[^}]*$',  # Unclosed brace
        r'\[[^\]]*$',  # Unclosed bracket
        r'\([^)]*$',  # Unclosed parenthesis
    ],
    TruncationType.MID_LIST: [
        r'\n\d+\.\s*$',  # Ends with numbered list marker
        r'\n[-*]\s*$',  # Ends with bullet marker
        r'\n\d+\.\s+\w+[^.!?]*$',  # Incomplete list item
    ],
    TruncationType.INCOMPLETE_THOUGHT: [
        r'(?:First|Second|Third|Fourth|Fifth|1\)|2\)|3\)|Step \d)[^.!?]*$',  # Incomplete step
        r'(?:However|Therefore|Additionally|Furthermore|Moreover)[^.!?]*$',  # Incomplete transition
        r'(?:For example|Such as|Including)[^.!?]*$',  # Incomplete example
    ],
}

CLEAN_END_PATTERNS = [
    r'[.!?]\s*$',  # Ends with proper punctuation
    r'```\s*$',  # Properly closed code block
    r'\n\s*$',  # Ends with newline after content
    r'(?:Let me know|Hope this helps|Feel free|Good luck)[^.]*[.!]\s*$',  # Common closings
]


def detect_truncation(text: str) -> TruncationType:
    """Analyze text to detect if it was truncated and how."""
    if not text or len(text.strip()) < 10:
        return TruncationType.UNKNOWN

    text = text.strip()

    # Check for clean endings first
    for pattern in CLEAN_END_PATTERNS:
        if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
            return TruncationType.CLEAN_END

    # Check for truncation patterns
    for truncation_type, patterns in TRUNCATION_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, text, re.MULTILINE | re.IGNORECASE):
                return truncation_type

    # Additional heuristics
    lines = text.split('\n')
    last_line = lines[-1].strip() if lines else ""

    # Check for incomplete sentences
    if last_line and not re.search(r'[.!?:;]$', last_line):
        if len(last_line) > 20:
            return TruncationType.MID_SENTENCE

    # Check for code blocks
    code_block_opens = text.count('```')
    if code_block_opens % 2 != 0:
        return TruncationType.MID_CODE_BLOCK

    return TruncationType.UNKNOWN


def calculate_completion_confidence(text: str) -> float:
    """Calculate confidence that a response is complete (0-1)."""
    if not text:
        return 0.0

    score = 0.5  # Base score
    text = text.strip()

    # Positive indicators
    if re.search(r'[.!?]\s*$', text):
        score += 0.15
    if re.search(r'(?:Let me know|Hope this helps|Feel free|Good luck)', text, re.IGNORECASE):
        score += 0.15
    if text.count('```') % 2 == 0:  # Balanced code blocks
        score += 0.1
    if len(text) > 500:  # Substantial response
        score += 0.05

    # Negative indicators
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


# ============================================================================
# Reformulation Engine
# ============================================================================

REFORMULATION_STRATEGIES = [
    {
        "name": "simplify",
        "description": "Break down into smaller parts",
        "template": "Let me approach this differently. Instead of answering everything at once, let's focus on the most important part first:\n\n{focus_question}\n\nPlease provide a complete answer to just this part."
    },
    {
        "name": "chunk",
        "description": "Request chunked responses",
        "template": "Please answer this question in clearly marked parts. After each part, pause and ask if I want you to continue.\n\nQuestion: {question}\n\nStart with Part 1."
    },
    {
        "name": "outline_first",
        "description": "Get outline then expand",
        "template": "For this question, please first provide a brief outline of your answer (just the main points), then I'll ask you to expand on specific parts:\n\n{question}"
    },
    {
        "name": "specific_format",
        "description": "Request specific format",
        "template": "Please answer the following question. Keep your response concise and complete. If you need more space, end with '[CONTINUED]' and I'll ask for the rest.\n\n{question}"
    },
    {
        "name": "reverse_approach",
        "description": "Ask for conclusion first",
        "template": "For this question, please start with your conclusion/answer first, then provide supporting details:\n\n{question}"
    },
]


def generate_reformulation(original_question: str, previous_attempts: List[str], truncation_type: TruncationType) -> Dict[str, str]:
    """Generate a reformulated version of the question based on what failed."""
    used_strategies = set()
    for attempt in previous_attempts:
        for strategy in REFORMULATION_STRATEGIES:
            if strategy["name"] in attempt:
                used_strategies.add(strategy["name"])

    # Select strategy based on truncation type and previous attempts
    strategy = None

    if truncation_type == TruncationType.MID_CODE_BLOCK:
        preferred = ["chunk", "outline_first"]
    elif truncation_type == TruncationType.MID_LIST:
        preferred = ["chunk", "specific_format"]
    elif truncation_type in [TruncationType.MID_SENTENCE, TruncationType.MID_WORD]:
        preferred = ["simplify", "chunk", "specific_format"]
    else:
        preferred = ["outline_first", "simplify", "reverse_approach"]

    # Find first unused strategy
    for pref in preferred:
        if pref not in used_strategies:
            strategy = next((s for s in REFORMULATION_STRATEGIES if s["name"] == pref), None)
            if strategy:
                break

    # Fallback to any unused strategy
    if not strategy:
        for s in REFORMULATION_STRATEGIES:
            if s["name"] not in used_strategies:
                strategy = s
                break

    # Last resort: use simplify with modification
    if not strategy:
        strategy = REFORMULATION_STRATEGIES[0]

    # Generate focus question for simplify strategy
    focus_question = original_question
    if "focus_question" in strategy["template"]:
        # Extract key question words
        if "?" in original_question:
            focus_question = original_question.split("?")[0] + "?"
        else:
            words = original_question.split()
            focus_question = " ".join(words[:min(20, len(words))]) + "..."

    reformulated = strategy["template"].format(
        question=original_question,
        focus_question=focus_question
    )

    return {
        "strategy": strategy["name"],
        "reformulated_question": reformulated,
        "explanation": strategy["description"]
    }


# ============================================================================
# MCP Tools - Input Models
# ============================================================================

class StartTaskInput(BaseModel):
    """Input for starting a new tracked task."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    question: str = Field(
        ...,
        description="The original question/task being worked on",
        min_length=5,
        max_length=10000
    )
    expected_completion_indicators: Optional[List[str]] = Field(
        default=None,
        description="Optional phrases that indicate completion (e.g., 'In conclusion', 'To summarize')"
    )


class CheckpointInput(BaseModel):
    """Input for saving a checkpoint."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    task_id: str = Field(..., description="The task ID to checkpoint")
    progress_description: str = Field(
        ...,
        description="Description of current progress",
        min_length=1,
        max_length=5000
    )
    completion_percentage: Optional[int] = Field(
        default=None,
        description="Estimated completion percentage (0-100)",
        ge=0,
        le=100
    )


class CheckCompletionInput(BaseModel):
    """Input for checking if a response was complete."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    task_id: Optional[str] = Field(
        default=None,
        description="Specific task ID to check, or None for the most recent"
    )
    response_text: str = Field(
        ...,
        description="The response text to analyze for completeness",
        min_length=1
    )


class GetRecoveryPlanInput(BaseModel):
    """Input for getting a recovery plan."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    task_id: Optional[str] = Field(
        default=None,
        description="Specific task ID, or None for most recent incomplete task"
    )


class MarkCompleteInput(BaseModel):
    """Input for marking a task complete."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    task_id: str = Field(..., description="The task ID to mark complete")
    final_notes: Optional[str] = Field(
        default=None,
        description="Optional notes about the completion"
    )


class ListTasksInput(BaseModel):
    """Input for listing tasks."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    status_filter: Optional[TaskStatus] = Field(
        default=None,
        description="Filter by status"
    )
    limit: int = Field(
        default=10,
        description="Maximum number of tasks to return",
        ge=1,
        le=50
    )


# ============================================================================
# MCP Tools
# ============================================================================

class AutoStartInput(BaseModel):
    """Input for auto-starting watchdog."""
    model_config = ConfigDict(str_strip_whitespace=True, extra='forbid')

    question: Optional[str] = Field(
        default=None,
        description="The question/task being worked on (optional - will auto-detect)"
    )


@mcp.tool(
    name="watchdog_auto_activate",
    annotations={
        "title": "Auto-Activate Watchdog (Always On)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def auto_activate(params: AutoStartInput = None) -> str:
    """AUTOMATICALLY activates the watchdog - THIS RUNS NO MATTER WHAT.

    This tool ensures the watchdog is ALWAYS active for every conversation.
    Call this at the start of ANY interaction to guarantee monitoring.

    The watchdog will:
    - Auto-create a tracking task if none exists
    - Monitor for truncation
    - Enable recovery if responses get cut off

    Args:
        params: Optional question to track

    Returns:
        JSON with active task info
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_auto_activate")
    question = params.question if params else None
    log.info(f"Question provided: {bool(question)}")
    if question:
        log.info(f"Question preview: {question[:100]}...")

    task = get_or_create_auto_task(question)

    if task:
        log.info(f"Watchdog ACTIVATED - Task ID: {task.task_id}")
        return json.dumps({
            "success": True,
            "watchdog_status": "ACTIVE",
            "mode": "always_on",
            "task_id": task.task_id,
            "message": "Watchdog is now monitoring this conversation. Truncation detection enabled.",
            "auto_features": [
                "Truncation detection",
                "Progress checkpointing",
                "Auto-recovery planning",
                "Response completeness analysis"
            ]
        }, indent=2)
    else:
        log.error("Failed to activate watchdog!")
        return json.dumps({
            "success": False,
            "error": "Could not activate watchdog"
        }, indent=2)


@mcp.tool(
    name="watchdog_start_task",
    annotations={
        "title": "Start Tracking a Task",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False
    }
)
async def start_task(params: StartTaskInput) -> str:
    """Start tracking a new task/question for completion monitoring.

    Call this at the beginning of a complex question or task to enable
    automatic detection of truncation and recovery assistance.

    Args:
        params: Contains the question to track and optional completion indicators

    Returns:
        JSON with task_id and confirmation message
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_start_task")
    log.info(f"Question: {params.question[:100]}...")

    tasks = load_tasks()

    task_id = generate_task_id(params.question)
    now = datetime.now().isoformat()

    task = TrackedTask(
        task_id=task_id,
        original_question=params.question,
        status=TaskStatus.IN_PROGRESS,
        created_at=now,
        updated_at=now,
        checkpoints=[],
        completion_indicators=params.expected_completion_indicators or []
    )

    tasks[task_id] = task
    save_tasks(tasks)

    log.info(f"Task STARTED: {task_id}")
    log.info(f"Completion indicators: {params.expected_completion_indicators}")

    return json.dumps({
        "success": True,
        "task_id": task_id,
        "message": f"Now tracking task. Use task_id '{task_id}' for checkpoints and completion checks.",
        "tip": "Save checkpoints during long responses with watchdog_checkpoint"
    }, indent=2)


@mcp.tool(
    name="watchdog_checkpoint",
    annotations={
        "title": "Save Progress Checkpoint",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def save_checkpoint(params: CheckpointInput) -> str:
    """Save a progress checkpoint for a tracked task.

    Call this periodically during long responses to save progress state.
    If the response gets truncated, this checkpoint will help resume.

    Args:
        params: Contains task_id, progress description, and optional completion %

    Returns:
        JSON with checkpoint confirmation
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_checkpoint")
    log.info(f"Task ID: {params.task_id}")
    log.info(f"Progress: {params.progress_description[:100]}...")
    log.info(f"Completion %: {params.completion_percentage}")

    tasks = load_tasks()

    if params.task_id not in tasks:
        log.warning(f"Task not found: {params.task_id}")
        return json.dumps({
            "success": False,
            "error": f"Task '{params.task_id}' not found. Start a task first with watchdog_start_task"
        }, indent=2)

    task = tasks[params.task_id]
    now = datetime.now().isoformat()

    checkpoint = {
        "timestamp": now,
        "description": params.progress_description,
        "completion_percentage": params.completion_percentage,
        "checkpoint_number": len(task.checkpoints) + 1
    }

    task.checkpoints.append(checkpoint)
    task.updated_at = now
    tasks[params.task_id] = task
    save_tasks(tasks)

    log.info(f"CHECKPOINT SAVED: #{checkpoint['checkpoint_number']} for task {params.task_id}")

    return json.dumps({
        "success": True,
        "checkpoint_number": checkpoint["checkpoint_number"],
        "message": "Progress saved. If interrupted, recovery will resume from here."
    }, indent=2)


@mcp.tool(
    name="watchdog_check_completion",
    annotations={
        "title": "Check Response Completeness",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def check_completion(params: CheckCompletionInput) -> str:
    """Analyze a response to determine if it completed successfully or was truncated.

    This tool performs sophisticated analysis of the response text to detect
    various types of truncation (mid-sentence, mid-code-block, etc.).

    NOTE: Watchdog auto-activates if not already running.

    Args:
        params: Contains optional task_id and the response text to analyze

    Returns:
        JSON with completion analysis, truncation type, and confidence score
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_check_completion")
    log.info(f"Task ID provided: {params.task_id}")
    log.info(f"Response text length: {len(params.response_text)} chars")

    # Ensure watchdog is active (always-on mode)
    ensure_watchdog_active()

    tasks = load_tasks()

    # Find the task
    task = None
    task_id = params.task_id

    if task_id and task_id in tasks:
        task = tasks[task_id]
        log.debug(f"Found specified task: {task_id}")
    elif not task_id:
        # Find most recent in-progress task
        in_progress = [t for t in tasks.values() if t.status == TaskStatus.IN_PROGRESS]
        if in_progress:
            task = max(in_progress, key=lambda t: t.updated_at)
            task_id = task.task_id
            log.debug(f"Using most recent in-progress task: {task_id}")

    # Analyze the response
    truncation_type = detect_truncation(params.response_text)
    confidence = calculate_completion_confidence(params.response_text)

    is_complete = truncation_type == TruncationType.CLEAN_END and confidence >= 0.7

    log.info(f"ANALYSIS RESULT: complete={is_complete}, truncation={truncation_type.value}, confidence={confidence:.2f}")

    # Update task if we have one
    if task:
        now = datetime.now().isoformat()
        task.last_response_snippet = params.response_text[-500:] if len(params.response_text) > 500 else params.response_text
        task.truncation_type = truncation_type
        task.updated_at = now

        if is_complete:
            task.status = TaskStatus.COMPLETED
            log.info(f"Task {task_id} marked as COMPLETED")
        elif truncation_type != TruncationType.CLEAN_END:
            task.status = TaskStatus.TRUNCATED
            log.warning(f"Task {task_id} marked as TRUNCATED ({truncation_type.value})")

        tasks[task_id] = task
        save_tasks(tasks)

    result = {
        "is_complete": is_complete,
        "confidence": round(confidence, 2),
        "truncation_type": truncation_type.value,
        "analysis": {
            "response_length": len(params.response_text),
            "ends_with_punctuation": bool(re.search(r'[.!?]\s*$', params.response_text.strip())),
            "has_balanced_code_blocks": params.response_text.count('```') % 2 == 0,
        }
    }

    if not is_complete:
        result["recommendation"] = "Use watchdog_get_recovery_plan to get continuation strategy"
        if truncation_type != TruncationType.CLEAN_END:
            result["detected_issue"] = f"Response appears to have been cut off ({truncation_type.value})"

    if task_id:
        result["task_id"] = task_id

    return json.dumps(result, indent=2)


@mcp.tool(
    name="watchdog_get_recovery_plan",
    annotations={
        "title": "Get Recovery/Continuation Plan",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_recovery_plan(params: GetRecoveryPlanInput) -> str:
    """Get a recovery plan for an incomplete or truncated task.

    Analyzes what went wrong and provides a reformulated approach to
    complete the task successfully.

    NOTE: Watchdog auto-activates if not already running.

    Args:
        params: Contains optional task_id (uses most recent incomplete if not specified)

    Returns:
        JSON with recovery plan including reformulated question and strategy
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_get_recovery_plan")
    log.info(f"Task ID provided: {params.task_id}")

    # Ensure watchdog is active (always-on mode)
    ensure_watchdog_active()

    tasks = load_tasks()

    # Find the task
    task = None

    if params.task_id and params.task_id in tasks:
        task = tasks[params.task_id]
        log.debug(f"Found specified task: {params.task_id}")
    else:
        # Find most recent incomplete task
        incomplete = [t for t in tasks.values()
                     if t.status in [TaskStatus.TRUNCATED, TaskStatus.NEEDS_CONTINUATION, TaskStatus.IN_PROGRESS]]
        if incomplete:
            task = max(incomplete, key=lambda t: t.updated_at)
            log.debug(f"Using most recent incomplete task: {task.task_id}")

    if not task:
        log.warning("No incomplete tasks found for recovery")
        return json.dumps({
            "success": False,
            "error": "No incomplete tasks found. All tasks are either completed or none exist.",
            "suggestion": "Start a new task with watchdog_start_task if you have a question to track"
        }, indent=2)

    # Generate recovery plan
    truncation_type = task.truncation_type or TruncationType.UNKNOWN
    reformulation = generate_reformulation(
        task.original_question,
        task.reformulations,
        truncation_type
    )

    log.info(f"RECOVERY PLAN generated for task {task.task_id}")
    log.info(f"Strategy: {reformulation['strategy']}")
    log.info(f"Truncation type: {truncation_type.value if truncation_type else 'unknown'}")

    # Build continuation prompt if we have checkpoints
    continuation_context = ""
    if task.checkpoints:
        last_checkpoint = task.checkpoints[-1]
        continuation_context = f"\n\nLast checkpoint ({last_checkpoint['timestamp']}):\n{last_checkpoint['description']}"

    recovery_plan = {
        "task_id": task.task_id,
        "original_question": task.original_question,
        "truncation_type": truncation_type.value if truncation_type else "unknown",
        "retry_count": task.retry_count,
        "recovery_strategy": {
            "name": reformulation["strategy"],
            "explanation": reformulation["explanation"],
            "reformulated_question": reformulation["reformulated_question"]
        },
        "last_progress": continuation_context if continuation_context else "No checkpoints saved",
        "instructions": [
            f"1. Use the reformulated question below (strategy: {reformulation['strategy']})",
            "2. Call watchdog_start_task with the new question to track it",
            "3. Save checkpoints with watchdog_checkpoint during long responses",
            "4. Check completion with watchdog_check_completion when done"
        ]
    }

    # Update task
    task.reformulations.append(f"{reformulation['strategy']}: {reformulation['reformulated_question'][:100]}...")
    task.retry_count += 1
    task.status = TaskStatus.NEEDS_CONTINUATION
    task.updated_at = datetime.now().isoformat()
    tasks[task.task_id] = task
    save_tasks(tasks)

    log.info(f"Task {task.task_id} updated: retry_count={task.retry_count}, status=NEEDS_CONTINUATION")

    return json.dumps(recovery_plan, indent=2)


@mcp.tool(
    name="watchdog_mark_complete",
    annotations={
        "title": "Mark Task as Complete",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def mark_complete(params: MarkCompleteInput) -> str:
    """Manually mark a task as complete.

    Use this when you've successfully completed a task and want to
    close it out in the tracking system.

    Args:
        params: Contains task_id and optional final notes

    Returns:
        JSON with confirmation
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_mark_complete")
    log.info(f"Task ID: {params.task_id}")
    log.info(f"Final notes: {params.final_notes[:100] if params.final_notes else 'None'}...")

    tasks = load_tasks()

    if params.task_id not in tasks:
        log.warning(f"Task not found: {params.task_id}")
        return json.dumps({
            "success": False,
            "error": f"Task '{params.task_id}' not found"
        }, indent=2)

    task = tasks[params.task_id]
    now = datetime.now().isoformat()

    task.status = TaskStatus.COMPLETED
    task.updated_at = now

    if params.final_notes:
        task.checkpoints.append({
            "timestamp": now,
            "description": f"COMPLETED: {params.final_notes}",
            "completion_percentage": 100
        })

    tasks[params.task_id] = task
    save_tasks(tasks)

    # Add to history
    history = load_history()
    history.append({
        "task_id": task.task_id,
        "question_preview": task.original_question[:100] + "...",
        "status": "completed",
        "retry_count": task.retry_count,
        "completed_at": now
    })
    save_history(history)

    log.info(f"Task COMPLETED: {params.task_id}")
    log.info(f"Total checkpoints: {len(task.checkpoints)}, Retry count: {task.retry_count}")

    return json.dumps({
        "success": True,
        "message": f"Task '{params.task_id}' marked as complete",
        "retry_count": task.retry_count,
        "total_checkpoints": len(task.checkpoints)
    }, indent=2)


@mcp.tool(
    name="watchdog_list_tasks",
    annotations={
        "title": "List Tracked Tasks",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def list_tasks(params: ListTasksInput) -> str:
    """List all tracked tasks with optional status filter.

    NOTE: Watchdog auto-activates if not already running.

    Args:
        params: Contains optional status filter and limit

    Returns:
        JSON with list of tasks
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_list_tasks")
    log.info(f"Status filter: {params.status_filter}")
    log.info(f"Limit: {params.limit}")

    # Ensure watchdog is active (always-on mode)
    ensure_watchdog_active()

    tasks = load_tasks()

    # Filter tasks
    filtered = list(tasks.values())
    if params.status_filter:
        filtered = [t for t in filtered if t.status == params.status_filter]

    # Sort by updated_at descending
    filtered.sort(key=lambda t: t.updated_at, reverse=True)

    # Limit results
    filtered = filtered[:params.limit]

    task_list = []
    for task in filtered:
        task_list.append({
            "task_id": task.task_id,
            "status": task.status.value,
            "question_preview": task.original_question[:80] + "..." if len(task.original_question) > 80 else task.original_question,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "checkpoint_count": len(task.checkpoints),
            "retry_count": task.retry_count
        })

    log.info(f"Listed {len(task_list)} tasks (total: {len(tasks)})")

    return json.dumps({
        "total_tasks": len(tasks),
        "filtered_count": len(task_list),
        "tasks": task_list
    }, indent=2)


@mcp.tool(
    name="watchdog_analyze_text",
    annotations={
        "title": "Analyze Text for Truncation",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def analyze_text(response_text: str) -> str:
    """Quick analysis of any text to check for truncation signs.

    A lightweight version of check_completion that doesn't require
    a tracked task. Good for ad-hoc analysis.

    NOTE: Watchdog auto-activates if not already running.

    Args:
        response_text: The text to analyze

    Returns:
        JSON with truncation analysis
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_analyze_text")
    log.info(f"Text length: {len(response_text)} chars")

    # Ensure watchdog is active (always-on mode)
    ensure_watchdog_active()

    truncation_type = detect_truncation(response_text)
    confidence = calculate_completion_confidence(response_text)

    is_complete = truncation_type == TruncationType.CLEAN_END and confidence >= 0.7
    log.info(f"ANALYSIS: complete={is_complete}, truncation={truncation_type.value}, confidence={confidence:.2f}")

    return json.dumps({
        "truncation_type": truncation_type.value,
        "completion_confidence": round(confidence, 2),
        "is_likely_complete": is_complete,
        "text_length": len(response_text),
        "analysis_details": {
            "ends_properly": bool(re.search(r'[.!?]\s*$', response_text.strip())),
            "balanced_code_blocks": response_text.count('```') % 2 == 0,
            "balanced_brackets": response_text.count('[') == response_text.count(']'),
            "balanced_braces": response_text.count('{') == response_text.count('}')
        }
    }, indent=2)


@mcp.tool(
    name="watchdog_get_status",
    annotations={
        "title": "Get Watchdog Status",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def get_status() -> str:
    """Get overall status of the watchdog system.

    This also ensures the watchdog is active (always-on mode).

    Returns summary statistics and any tasks needing attention.

    Returns:
        JSON with system status
    """
    log.info("=" * 40)
    log.info("TOOL CALLED: watchdog_get_status")

    # Ensure watchdog is active (always-on mode)
    active_task = ensure_watchdog_active()

    tasks = load_tasks()
    history = load_history()

    status_counts = {}
    for task in tasks.values():
        status_counts[task.status.value] = status_counts.get(task.status.value, 0) + 1

    needs_attention = [
        {
            "task_id": t.task_id,
            "status": t.status.value,
            "question_preview": t.original_question[:50] + "..."
        }
        for t in tasks.values()
        if t.status in [TaskStatus.TRUNCATED, TaskStatus.NEEDS_CONTINUATION, TaskStatus.FAILED]
    ]

    log.info(f"STATUS: {len(tasks)} tasks, {len(needs_attention)} need attention")
    log.info(f"Active task: {active_task.task_id if active_task else 'None'}")
    log.info(f"Status breakdown: {status_counts}")

    return json.dumps({
        "system_status": "ACTIVE - ALWAYS ON",
        "always_on_mode": True,
        "auto_start_enabled": AUTO_START_ENABLED,
        "current_active_task": active_task.task_id if active_task else None,
        "data_directory": str(WATCHDOG_DATA_DIR),
        "log_file": str(LOG_FILE),
        "statistics": {
            "total_tracked_tasks": len(tasks),
            "completed_in_history": len([h for h in history if h.get("status") == "completed"]),
            "status_breakdown": status_counts
        },
        "needs_attention": needs_attention[:5],
        "available_tools": [
            "watchdog_auto_activate - Auto-activate watchdog (always on)",
            "watchdog_start_task - Begin tracking a new question",
            "watchdog_checkpoint - Save progress during long responses",
            "watchdog_check_completion - Analyze if response completed",
            "watchdog_get_recovery_plan - Get strategy for incomplete tasks",
            "watchdog_mark_complete - Manually mark task done",
            "watchdog_list_tasks - View all tracked tasks",
            "watchdog_analyze_text - Quick truncation check"
        ]
    }, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
