from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
import threading
import queue
import time

from .base_failure_detector import FailureDetectionModule


class AsyncFailureDetectionModule(FailureDetectionModule, ABC):
    """Abstract asynchronous failure detection module.

    This class provides a generic asynchronous processing framework that is
    agnostic to task input structures. Subclasses should implement
    `handle_async_task` to define how tasks are processed and what results are
    emitted.

    It intentionally does not implement the abstract methods of
    `FailureDetectionModule` so that this class remains abstract.
    """

    def __init__(self, max_queue_size: int = 2) -> None:
        self.async_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max_queue_size)
        self.async_result_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.async_thread_stop = threading.Event()
        self.async_thread: Optional[threading.Thread] = None

    @abstractmethod
    def handle_async_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a single asynchronous task.

        Subclasses must implement this to handle different `task_type`s
        and return an optional result dict to be enqueued into
        `async_result_queue`. Return None if there is nothing to emit.
        """
        raise NotImplementedError

    def start_async_processing(self) -> None:
        """Start the asynchronous processing thread."""
        if self.async_thread is not None and self.async_thread.is_alive():
            return
        self.async_thread_stop.clear()
        self.async_thread = threading.Thread(target=self._async_processing_thread, daemon=True)
        self.async_thread.start()

    def stop_async_processing(self) -> None:
        """Stop the asynchronous processing thread."""
        if self.async_thread is not None:
            self.async_thread_stop.set()
            self.async_thread.join(timeout=1.0)
            self.async_thread = None

    def _async_processing_thread(self) -> None:
        """Thread function for asynchronous processing loop."""
        while not self.async_thread_stop.is_set():
            try:
                task = self.async_queue.get(timeout=0.1)
                try:
                    result = self.handle_async_task(task)
                    if result is not None:
                        self.async_result_queue.put(result)
                finally:
                    # Always mark the task as done to avoid blocking joiners
                    try:
                        self.async_queue.task_done()
                    except Exception:
                        pass
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in async processing thread: {e}")
                import traceback
                traceback.print_exc()
                continue

    def submit_task(self, task: Dict[str, Any]) -> bool:
        """Submit a single task for asynchronous processing.

        Returns True on success, False if the queue is full.
        """
        try:
            self.async_queue.put_nowait(task)
            return True
        except queue.Full:
            return False

    def submit_tasks(self, tasks: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Submit multiple tasks. Returns (num_submitted, num_failed)."""
        submitted = 0
        failed = 0
        for task in tasks:
            if self.submit_task(task):
                submitted += 1
            else:
                failed += 1
        return submitted, failed

    def get_results(self) -> List[Dict[str, Any]]:
        """Get all available results from the result queue."""
        results: List[Dict[str, Any]] = []
        try:
            while not self.async_result_queue.empty():
                results.append(self.async_result_queue.get_nowait())
        except queue.Empty:
            pass
        return results

    def wait_for_final_results(self, **kwargs) -> Tuple[bool, str, int]:
        """Wait for all results to be processed"""
        raise NotImplementedError

    def empty_queue(self) -> None:
        """Empty the task queue."""
        while not self.async_queue.empty():
            try:
                self.async_queue.get_nowait()
            except queue.Empty:
                break

    def empty_result_queue(self) -> None:
        """Empty the result queue."""
        while not self.async_result_queue.empty():
            try:
                self.async_result_queue.get_nowait()
            except queue.Empty:
                break
