"""
Simplified recap generation module.

This module provides a streamlined approach to generating "Previously On" recaps
following the core LLM workflow without unnecessary complexity.
"""

from .recap_generator import RecapGenerator
from .models import RecapResult, Event, VideoClip

__all__ = ['RecapGenerator', 'RecapResult', 'Event', 'VideoClip']
