import signal
import resource
import logging

logger = logging.getLogger(__name__)

def run_with_limits(fn, *, timeout_s: float, mem_mb: int):
    """Run function with resource limits.

    Args:
        fn: Function to run.
        timeout_s: Timeout in seconds.
        mem_mb: Memory limit in MB.

    Returns:
        Result or None if failed.
    """
    # Stub implementation for POSIX
    try:
        # Set limits
        resource.setrlimit(resource.RLIMIT_CPU, (int(timeout_s), int(timeout_s)))
        resource.setrlimit(resource.RLIMIT_AS, (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024))
        return fn()
    except:
        return None