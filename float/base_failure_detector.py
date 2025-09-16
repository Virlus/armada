from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import torch

class FailureDetectionModule(ABC):
    """Abstract base class for failure detection modules"""
    
    @abstractmethod
    def runtime_initialize(self, **kwargs):
        """Initialize the failure detection module at runtime"""
        pass
    
    @abstractmethod
    def detect_failure(self, **kwargs) -> Tuple[bool, Optional[str]]:
        """Detect if failure occurred. Returns (failure_flag, failure_reason)"""
        pass
    
    @abstractmethod
    def process_step(self, step_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single step and return any additional data"""
        pass
    
    @abstractmethod
    def finalize_episode(self, episode_data: Dict[str, Any]) -> Dict[str, Any]:
        """Finalize episode processing and return any additional data"""
        pass