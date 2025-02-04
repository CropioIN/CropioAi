"""
CropioAI Flow Persistence.

This module provides interfaces and implementations for persisting flow states.
"""

from typing import Any, Dict, TypeVar, Union

from pydantic import BaseModel

from cropioai.flow.persistence.base import FlowPersistence
from cropioai.flow.persistence.decorators import persist
from cropioai.flow.persistence.sqlite import SQLiteFlowPersistence

__all__ = ["FlowPersistence", "persist", "SQLiteFlowPersistence"]

StateType = TypeVar('StateType', bound=Union[Dict[str, Any], BaseModel])
DictStateType = Dict[str, Any]
