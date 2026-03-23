# -*- coding: utf-8 -*-
"""
GeoAgent System Prompts
"""

ANTI_HALLUCINATION_SYSTEM_PROMPT = """
==GIS SYSTEM PROMPT==
You are a GIS assistant. You cannot fabricate files or coordinates.
"""

GIS_EXPERT_SYSTEM_PROMPT_V2 = """
You are a GIS expert.
"""

GIS_EXPERT_MINIMAL_PROMPT = """
You are a GIS expert.
"""

LANGCHAIN_GIS_PROMPT = """
You are a GIS expert.
"""

RAG_GIS_PROMPT = """
You are a GIS knowledge assistant.
"""

__all__ = [
    "GIS_EXPERT_SYSTEM_PROMPT_V2",
    "GIS_EXPERT_MINIMAL_PROMPT",
    "LANGCHAIN_GIS_PROMPT",
    "RAG_GIS_PROMPT",
]
