"""Process and Resource Lifecycle Management for Disaggregated KV Transport.

Provides hardened lifecycle utilities to guarantee bounded cleanup,
kill escalation, zombie prevention, and objective verification of child termination.
"""

import os
import signal
import time
import multiprocessing as mp
from typing import Dict, Any, List, Optional, Tuple


def verify_pid_dead(pid: int) -> bool:
    """Verifies that an OS process ID is no longer alive."""
    if pid <= 0:
        return True
    try:
        # Signal 0 does not terminate the process; it only performs error checking
        os.kill(pid, 0)
        return False
    except ProcessLookupError:
        return True
    except PermissionError:
        # Process exists but owned by another user
        return False
    except OSError:
        return True


def safe_cleanup_processes(
    processes: List[Tuple[Optional[mp.Process], Optional[int]]],
    temp_files: Optional[List[str]] = None,
    queues: Optional[List[mp.Queue]] = None,
    timeout_terminate: float = 2.0,
    timeout_kill: float = 1.0,
) -> Dict[str, Any]:
    """
    Guarantees deterministic cleanup of spawned child processes:
    1. Checks every started process, requests graceful termination (terminate).
    2. Bounded join; escalates to SIGKILL if still alive.
    3. Verifies process termination using OS signals (os.kill(pid, 0)).
    4. Drains/closes multiprocessing queues and unlinks temporary files.
    5. Returns an audited report of cleanup results.
    """
    surviving_pids: List[int] = []
    reaped_pids: List[int] = []

    for proc, recorded_pid in processes:
        pid = proc.pid if (proc is not None and proc.pid is not None) else recorded_pid
        if proc is not None:
            try:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=timeout_terminate)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=timeout_kill)
            except Exception:
                pass

        if pid is not None:
            if not verify_pid_dead(pid):
                # Escalate via direct OS kill if still surviving
                try:
                    os.kill(pid, signal.SIGKILL)
                    time.sleep(0.1)
                except Exception:
                    pass
                if not verify_pid_dead(pid):
                    surviving_pids.append(pid)
                else:
                    reaped_pids.append(pid)
            else:
                reaped_pids.append(pid)

    # Clean queues
    if queues:
        for q in queues:
            try:
                q.close()
                q.cancel_join_thread()
            except Exception:
                pass

    # Clean temp files
    leaked_files: List[str] = []
    deleted_files: List[str] = []
    if temp_files:
        for path in temp_files:
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                    deleted_files.append(path)
                except Exception:
                    leaked_files.append(path)

    all_reaped = len(surviving_pids) == 0
    all_deleted = len(leaked_files) == 0

    return {
        "all_child_processes_reaped": all_reaped,
        "temporary_files_deleted": all_deleted,
        "reaped_pids": reaped_pids,
        "surviving_pids": surviving_pids,
        "deleted_temp_files": deleted_files,
        "leaked_temp_files": leaked_files,
    }


def validate_cleanup_report(report: Any) -> bool:
    """Validates that a process/resource cleanup report exists, is non-empty, and confirms complete cleanup."""
    if not isinstance(report, dict) or not report:
        return False
    if report.get("all_child_processes_reaped") is not True:
        return False
    if report.get("temporary_files_deleted") is not True:
        return False
    if report.get("surviving_pids") != []:
        return False
    if report.get("leaked_temp_files") != []:
        return False
    return True
