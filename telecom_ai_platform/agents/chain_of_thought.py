"""
Chain of Thought Agent for Telecom AI Platform

This module implements advanced chain of thought reasoning for complex
telecom network analysis with step-by-step logical processing.
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple, Callable
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import uuid
from enum import Enum

import google.generativeai as genai
from langchain.schema import BaseMessage, HumanMessage, AIMessage, SystemMessage
try:
    from langchain_google_genai import ChatGoogleGenerativeAI  # optional; we will fallback if incompatible
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None

from ..core.config import TelecomConfig
from ..core.data_processor import TelecomDataProcessor
from ..models.anomaly_detector import KPIAnomalyDetector
from ..utils.logger import LoggerMixin, log_function_call
from .conversational_ai import NetworkAnalysisTools


class ReasoningType(Enum):
    """Types of reasoning steps"""
    PROBLEM_DECOMPOSITION = "problem_decomposition"
    DATA_ANALYSIS = "data_analysis"
    PATTERN_RECOGNITION = "pattern_recognition"
    CAUSAL_INFERENCE = "causal_inference"
    HYPOTHESIS_FORMATION = "hypothesis_formation"
    EVIDENCE_EVALUATION = "evidence_evaluation"
    CONCLUSION_SYNTHESIS = "conclusion_synthesis"
    RECOMMENDATION_GENERATION = "recommendation_generation"


@dataclass
class ReasoningStep:
    """Represents a step in the chain of thought reasoning process"""
    step_id: str
    step_number: int
    reasoning_type: ReasoningType
    question: str
    analysis: str
    findings: List[str]
    evidence: Optional[Dict] = None
    confidence: float = 0.0
    timestamp: datetime = None
    duration_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'step_id': self.step_id,
            'step_number': self.step_number,
            'reasoning_type': self.reasoning_type.value,
            'question': self.question,
            'analysis': self.analysis,
            'findings': self.findings,
            'evidence': self.evidence or {},
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'duration_ms': self.duration_ms
        }


@dataclass
class ChainOfThoughtResult:
    """Complete chain of thought analysis result"""
    query: str
    reasoning_chain: List[ReasoningStep]
    final_conclusion: str
    key_insights: List[str]
    recommendations: List[str]
    confidence: float
    total_processing_time_ms: float
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            'query': self.query,
            'reasoning_chain': [step.to_dict() for step in self.reasoning_chain],
            'final_conclusion': self.final_conclusion,
            'key_insights': self.key_insights,
            'recommendations': self.recommendations,
            'confidence': self.confidence,
            'total_processing_time_ms': self.total_processing_time_ms,
            'timestamp': self.timestamp.isoformat()
        }


class ChainOfThoughtAgent(LoggerMixin):
    """
    Advanced Chain of Thought Agent for complex telecom network analysis.
    
    This agent breaks down complex problems into logical steps, analyzes
    each step systematically, and builds comprehensive conclusions through
    structured reasoning.
    """
    
    def __init__(self, config: TelecomConfig):
        """Initialize the Chain of Thought Agent"""
        self.config = config
        self.data_processor = TelecomDataProcessor(config)
        self.anomaly_detector = KPIAnomalyDetector(config)
        self.network_tools = NetworkAnalysisTools(config, self.data_processor, self.anomaly_detector)
        
        # Initialize LLM
        self._initialize_llm()
        
        # Reasoning templates
        self._setup_reasoning_templates()
        
        # Load pre-trained anomaly models if available (enables concrete, per-KPI analysis)
        try:
            self.anomaly_detector.load_all_models()
        except Exception as _e:
            # Non-fatal; we'll train on the fly if needed
            self.logger.warning(f"Could not load pre-trained models: {_e}")

        self.logger.info("Chain of Thought Agent initialized successfully")
    
    def _initialize_llm(self):
        """Initialize the Google Gemini LLM"""
        try:
            genai.configure(api_key=self.config.gemini_api_key)
            # Prefer direct GenerativeModel to avoid wrapper version incompatibilities
            self._gm = genai.GenerativeModel(
                model_name=self.config.agent.model_name,
                generation_config={
                    "temperature": self.config.agent.temperature,
                    "max_output_tokens": self.config.agent.max_tokens,
                },
            )
            # LangChain wrapper (optional) kept for future use
            self.llm = None
            if ChatGoogleGenerativeAI is not None:
                try:
                    self.llm = ChatGoogleGenerativeAI(
                        model=self.config.agent.model_name,
                        temperature=self.config.agent.temperature,
                        max_tokens=self.config.agent.max_tokens,
                        google_api_key=self.config.gemini_api_key,
                    )
                except Exception:
                    # Non-fatal: we'll use direct client instead
                    self.llm = None
            
            self.logger.info("Chain of Thought LLM (direct client) initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize LLM: {e}")
            raise

    # ---------- Query classification and fast paths ----------
    def _classify_query(self, query: str) -> str:
        """Classify query into categories: greeting, definition, tool_analysis, general."""
        q = (query or "").strip().lower()
        if not q:
            return "general"
        # Greetings / small talk
        greetings = ["hi", "hello", "hey", "good morning", "good evening", "good afternoon"]
        if any(q == g or q.startswith(g + " ") for g in greetings):
            return "greeting"
        # Definition-style queries
        import re
        if re.search(r"^(what\s+is|what\s+does|define|meaning\s+of)\b", q):
            return "definition"
        # Tool-required analysis (anomalies, site/time filters, comparisons)
        tool_terms = ["anomal", "which kpi has", "last week", "site ", "detect", "highest", "per site", "per sector"]
        if any(term in q for term in tool_terms):
            return "tool_analysis"
        return "general"

    def _quick_greeting(self) -> str:
        return "Hello. How can I help with your telecom KPIs or anomalies today?"

    def _quick_define(self, query: str) -> str:
        """Provide a concise definition using the LLM without full chain-of-thought."""
        try:
            prompt = f"Provide a concise, one-paragraph technical definition: {query}. Keep it under 80 words."
            return self._gemini_generate_text(prompt, system_text="You are a concise telecom glossary.")
        except Exception:
            # Safe fallback
            return "A quick definition isn't available right now. Please try again or enable web search."

    def _gemini_generate_text(self, user_text: str, system_text: str) -> str:
        """Generate text using the direct google-generativeai client.
        Ensures the request contains 'contents' to satisfy the API contract.
        """
        try:
            # For gemini 1.5, pass system instruction on model construction if needed
            model = self._gm
            # Build contents explicitly (avoid wrapper conversions)
            contents = [
                {"role": "user", "parts": [user_text]}
            ]
            response = model.generate_content(contents)
            return getattr(response, "text", "") or ""
        except Exception as e:
            # Fallback to wrapper if available
            if self.llm is not None:
                try:
                    # Use simple prompt string with wrapper
                    return self.llm.invoke(user_text).content
                except Exception as e2:
                    raise RuntimeError(f"Gemini generate failed: {e2}") from e
            raise RuntimeError(f"Gemini generate failed: {e}")
    
    def _setup_reasoning_templates(self):
        """Setup reasoning templates for different types of analysis"""
        self.reasoning_templates = {
            ReasoningType.PROBLEM_DECOMPOSITION: {
                "prompt": "Break down this telecom network problem into smaller, analyzable components:",
                "questions": [
                    "What are the key components of this problem?",
                    "Which KPIs are potentially involved?",
                    "What timeframes should we consider?",
                    "Are there site-specific or network-wide implications?"
                ]
            },
            
            ReasoningType.DATA_ANALYSIS: {
                "prompt": "Analyze the available data to understand the current state:",
                "questions": [
                    "What does the current data tell us?",
                    "Are there any obvious patterns or anomalies?",
                    "What is the data quality and coverage?",
                    "What additional data might we need?"
                ]
            },
            
            ReasoningType.PATTERN_RECOGNITION: {
                "prompt": "Identify patterns and correlations in the network data:",
                "questions": [
                    "What patterns emerge from the data?",
                    "Are there correlations between different KPIs?",
                    "Do patterns vary by time, location, or other factors?",
                    "What do these patterns suggest about network behavior?"
                ]
            },
            
            ReasoningType.CAUSAL_INFERENCE: {
                "prompt": "Determine potential causes and relationships:",
                "questions": [
                    "What could be causing the observed behavior?",
                    "Are there upstream or downstream effects?",
                    "How do different network components interact?",
                    "What external factors might be involved?"
                ]
            },
            
            ReasoningType.HYPOTHESIS_FORMATION: {
                "prompt": "Form testable hypotheses about the network behavior:",
                "questions": [
                    "What are the most likely explanations?",
                    "How can we test these hypotheses?",
                    "What evidence would support or refute each hypothesis?",
                    "Which hypothesis has the strongest initial support?"
                ]
            },
            
            ReasoningType.EVIDENCE_EVALUATION: {
                "prompt": "Evaluate the available evidence for each hypothesis:",
                "questions": [
                    "What evidence supports each hypothesis?",
                    "How reliable and complete is the evidence?",
                    "Are there conflicting pieces of evidence?",
                    "What gaps exist in our evidence?"
                ]
            },
            
            ReasoningType.CONCLUSION_SYNTHESIS: {
                "prompt": "Synthesize findings into coherent conclusions:",
                "questions": [
                    "What can we conclude with high confidence?",
                    "What remains uncertain or requires further investigation?",
                    "How do our findings relate to network performance goals?",
                    "What are the broader implications?"
                ]
            },
            
            ReasoningType.RECOMMENDATION_GENERATION: {
                "prompt": "Generate actionable recommendations based on analysis:",
                "questions": [
                    "What immediate actions should be taken?",
                    "What longer-term improvements are needed?",
                    "How should recommendations be prioritized?",
                    "What are the potential risks and benefits of each recommendation?"
                ]
            }
        }
    
    @log_function_call
    async def analyze_with_chain_of_thought(
        self, 
        query: str, 
        context: Optional[Dict] = None,
        include_steps: Optional[List[ReasoningType]] = None,
        use_web_search: bool = False,
    use_mcp: bool = False,
    progress_cb: Optional[Callable[["ReasoningStep"], None]] = None
    ) -> ChainOfThoughtResult:
        """
        Perform comprehensive chain of thought analysis
        
        Args:
            query: The question or problem to analyze
            context: Optional context information
            include_steps: Optional list of reasoning steps to include
            
        Returns:
            ChainOfThoughtResult with complete reasoning chain
        """
        start_time = time.time()

        # Fast paths for simple queries or when tools are missing
        qclass = self._classify_query(query)
        if qclass == "greeting":
            final = self._quick_greeting()
            return ChainOfThoughtResult(
                query=query,
                reasoning_chain=[],
                final_conclusion=final,
                key_insights=[],
                recommendations=[],
                confidence=0.95,
                total_processing_time_ms=(time.time() - start_time) * 1000,
            )
        if qclass == "definition":
            final = self._quick_define(query)
            return ChainOfThoughtResult(
                query=query,
                reasoning_chain=[],
                final_conclusion=final,
                key_insights=[],
                recommendations=[],
                confidence=0.9,
                total_processing_time_ms=(time.time() - start_time) * 1000,
            )

        # If the query implies tool-based analysis, check tool availability first
        if qclass == "tool_analysis":
            data_ok = self.network_tools.current_data is not None
            models_ok = self.anomaly_detector.is_fitted
            if not data_ok:
                # Attempt to auto-load default dataset from known locations
                candidates = []
                try:
                    # Configured data_dir + data_file
                    dd = Path(self.config.data_dir)
                    candidates.append(dd / self.config.data.data_file)
                except Exception:
                    pass
                # Package data directory
                try:
                    pkg_data = Path(__file__).parent.parent / "data" / "AD_data_10KPI.csv"
                    candidates.append(pkg_data)
                except Exception:
                    pass
                # Repository root data (heuristic)
                try:
                    repo_root = Path(__file__).parent.parent.parent
                    candidates.append(repo_root / "data" / "AD_data_10KPI.csv")
                except Exception:
                    pass
                loaded_df = None
                for p in candidates:
                    try:
                        if p and p.exists():
                            df = pd.read_csv(p)
                            if "Date" in df.columns:
                                df["Date"] = pd.to_datetime(df["Date"])
                            loaded_df = df
                            break
                    except Exception:
                        continue
                if loaded_df is not None:
                    self.network_tools.current_data = loaded_df
                    data_ok = True
                if not data_ok:
                    final = (
                        "I can't perform that analysis right now because the required data is not available."
                    )
                    return ChainOfThoughtResult(
                        query=query,
                        reasoning_chain=[],
                        final_conclusion=final,
                        key_insights=[],
                        recommendations=["Load data into the agent (current_data) and retry the analysis."],
                        confidence=0.7,
                        total_processing_time_ms=(time.time() - start_time) * 1000,
                    )
            # If models are missing but data exists, continue using statistical fallback; annotate context
            if context is None:
                context = {}
            context["models_available"] = bool(models_ok)
            if not models_ok:
                context["model_fallback"] = "Anomaly models unavailable; using z-score statistical fallback"
        
        self.logger.info(f"Starting chain of thought analysis for: {query[:100]}...")
        
    # Enhance context with web search if requested
        if use_web_search and context is None:
            context = {}
            try:
                if hasattr(self.network_tools, 'search_web'):
                    self.logger.info("Using web search to enhance analysis...")
                    search_query = f"telecom network {query}"
                    search_results = await self.network_tools.search_web(search_query)
                    if search_results:
                        context['web_search'] = search_results
                        self.logger.info(f"Web search found {len(search_results)} results")
            except Exception as e:
                self.logger.error(f"Web search failed: {e}")
                context['web_search_error'] = str(e)
        
        # Enhance context with MCP data if requested
        if use_mcp and context is None:
            context = {}
            try:
                if hasattr(self.network_tools, 'use_mcp_tool'):
                    self.logger.info("Using MCP tools to enhance analysis...")
                    mcp_result = await self.network_tools.use_mcp_tool(
                        "telecom_analysis", {"query": query}
                    )
                    if mcp_result:
                        context['mcp_analysis'] = mcp_result
                        self.logger.info("Successfully retrieved MCP data")
            except Exception as e:
                self.logger.error(f"MCP integration failed: {e}")
                context['mcp_error'] = str(e)
        
        # Default reasoning steps for telecom network analysis
        if include_steps is None:
            include_steps = [
                ReasoningType.PROBLEM_DECOMPOSITION,
                ReasoningType.DATA_ANALYSIS,
                ReasoningType.PATTERN_RECOGNITION,
                ReasoningType.CAUSAL_INFERENCE,
                ReasoningType.HYPOTHESIS_FORMATION,
                ReasoningType.EVIDENCE_EVALUATION,
                ReasoningType.CONCLUSION_SYNTHESIS,
                ReasoningType.RECOMMENDATION_GENERATION
            ]
        
        reasoning_chain = []
        
        # Execute each reasoning step
        for step_number, reasoning_type in enumerate(include_steps, 1):
            step_start_time = time.time()
            
            reasoning_step = await self._execute_reasoning_step(
                step_number=step_number,
                reasoning_type=reasoning_type,
                query=query,
                context=context,
                previous_steps=reasoning_chain
            )
            
            reasoning_step.duration_ms = (time.time() - step_start_time) * 1000
            reasoning_chain.append(reasoning_step)
            
            self.logger.info(f"Completed reasoning step {step_number}: {reasoning_type.value}")
            # Stream progress to caller if callback provided
            try:
                if progress_cb is not None:
                    progress_cb(reasoning_step)
            except Exception as _cb_err:
                # Do not fail analysis due to UI callback errors
                self.logger.debug(f"Progress callback error (ignored): {_cb_err}")
        
        # Synthesize final conclusion
        final_conclusion = await self._synthesize_final_conclusion(query, reasoning_chain)
        
        # Extract key insights and recommendations
        key_insights = self._extract_key_insights(reasoning_chain)
        recommendations = self._extract_recommendations(reasoning_chain)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(reasoning_chain)
        
        total_time_ms = (time.time() - start_time) * 1000
        
        result = ChainOfThoughtResult(
            query=query,
            reasoning_chain=reasoning_chain,
            final_conclusion=final_conclusion,
            key_insights=key_insights,
            recommendations=recommendations,
            confidence=overall_confidence,
            total_processing_time_ms=total_time_ms
        )
        
        self.logger.info(f"Chain of thought analysis completed in {total_time_ms:.2f}ms")
        return result
    
    async def _execute_reasoning_step(
        self,
        step_number: int,
        reasoning_type: ReasoningType,
        query: str,
        context: Optional[Dict],
        previous_steps: List[ReasoningStep]
    ) -> ReasoningStep:
        """Execute a single reasoning step"""
        
        template = self.reasoning_templates[reasoning_type]
        
        # Build context from previous steps
        previous_context = ""
        if previous_steps:
            previous_context = "\n\nPrevious Analysis Steps:\n"
            for step in previous_steps[-3:]:  # Include last 3 steps for context
                previous_context += f"Step {step.step_number} ({step.reasoning_type.value}): {step.analysis[:200]}...\n"
        
        # Create specific prompt for this reasoning step
        step_prompt = f"""
You are conducting step {step_number} of a chain of thought analysis for a telecom network problem.

ORIGINAL QUERY: {query}

CURRENT REASONING STEP: {reasoning_type.value.replace('_', ' ').title()}
STEP OBJECTIVE: {template['prompt']}

KEY QUESTIONS TO ADDRESS:
{chr(10).join(f"- {q}" for q in template['questions'])}

{previous_context}

CONTEXT INFORMATION:
{json.dumps(context or {}, indent=2)}

NETWORK STATUS:
- Data Available: {'Yes' if self.network_tools.current_data is not None else 'No'}
- Models Loaded: {'Yes' if self.anomaly_detector.is_fitted else 'No'}
- Available KPIs: {', '.join(self.config.data.kpi_columns)}

Please provide a thorough analysis for this step, including:
1. Direct answers to the key questions
2. Specific findings based on available data
3. Your confidence level in these findings (0.0 to 1.0)
4. Any data or tools that should be used

Focus specifically on this step's objective. Be analytical and precise.
"""
        
        # Get LLM response
        try:
            # Use direct google-generativeai client to avoid wrapper incompatibilities
            analysis_text = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._gemini_generate_text(
                    user_text=step_prompt,
                    system_text="You are a senior telecom network analyst performing structured reasoning.",
                ),
            )
            
            # Parse the response to extract findings and confidence
            findings, confidence = self._parse_reasoning_response(analysis_text)
            
            # Gather evidence if this is a data analysis step
            evidence = None
            if reasoning_type in [ReasoningType.DATA_ANALYSIS, ReasoningType.EVIDENCE_EVALUATION]:
                evidence = await self._gather_evidence(query, previous_steps)
            
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                step_number=step_number,
                reasoning_type=reasoning_type,
                question=template['prompt'],
                analysis=analysis_text,
                findings=findings,
                evidence=evidence,
                confidence=confidence
            )
            
        except Exception as e:
            self.logger.error(f"Error in reasoning step {step_number}: {e}")
            
            return ReasoningStep(
                step_id=str(uuid.uuid4()),
                step_number=step_number,
                reasoning_type=reasoning_type,
                question=template['prompt'],
                analysis=f"Error in analysis: {str(e)}",
                findings=[f"Unable to complete analysis due to error: {str(e)}"],
                confidence=0.0
            )
    
    def _parse_reasoning_response(self, response_text: str) -> Tuple[List[str], float]:
        """Parse LLM response to extract findings and confidence"""
        findings = []
        confidence = 0.5  # Default confidence
        
        lines = response_text.split('\n')
        
        # Extract bullet points and numbered lists as findings
        for line in lines:
            line = line.strip()
            if line.startswith(('- ', '• ', '* ')) or (len(line) > 0 and line[0].isdigit() and '. ' in line):
                finding = line.lstrip('-•* ').split('. ', 1)[-1].strip()
                if len(finding) > 10:  # Filter out very short findings
                    findings.append(finding)
        
        # Look for confidence indicators
        confidence_indicators = ['confidence', 'certain', 'likely', 'probable']
        for line in lines:
            line_lower = line.lower()
            if any(indicator in line_lower for indicator in confidence_indicators):
                # Try to extract numeric confidence
                import re
                numbers = re.findall(r'(\d+(?:\.\d+)?)', line)
                if numbers:
                    try:
                        conf_value = float(numbers[-1])
                        if conf_value <= 1.0:
                            confidence = conf_value
                        elif conf_value <= 100:
                            confidence = conf_value / 100
                    except ValueError:
                        pass
        
        # If no findings extracted, use first few sentences
        if not findings:
            sentences = response_text.split('. ')
            findings = [s.strip() + '.' for s in sentences[:3] if len(s.strip()) > 20]
        
        return findings, confidence
    
    async def _gather_evidence(self, query: str, previous_steps: List[ReasoningStep]) -> Dict:
        """Gather evidence through data analysis tools"""
        evidence = {}
        
        try:
            # If we have data loaded, get basic statistics
            if self.network_tools.current_data is not None:
                data = self.network_tools.current_data
                evidence['data_summary'] = {
                    'total_records': len(data),
                    'date_range': f"{data['Date'].min()} to {data['Date'].max()}" if 'Date' in data.columns else "Unknown",
                    'sites_count': data['Site_ID'].nunique() if 'Site_ID' in data.columns else 0,
                    'kpis_available': [col for col in self.config.data.kpi_columns if col in data.columns]
                }

                # Ensure models are ready; load or train on the fly
                if not self.anomaly_detector.is_fitted:
                    try:
                        # Attempt quick load again (in case models became available)
                        self.anomaly_detector.load_all_models()
                    except Exception:
                        pass
                    # Respect non-debug preference: don't auto-train unless debug_mode
                    if (not self.anomaly_detector.is_fitted) and self.config.debug_mode:
                        self.anomaly_detector.fit(data)

                # Parse filters (site/time) from the user's query
                site_id, date_range = self._parse_query_filters(query, data)

                # Run anomaly detection with filters to generate concrete evidence
                if self.anomaly_detector.is_fitted:
                    results = self.anomaly_detector.detect_anomalies(
                        data,
                        kpi_name=None,
                        site_id=site_id,
                        date_range=date_range,
                    )
                    anomalies = [r for r in results if r.is_anomaly]
                    # Aggregate per KPI counts
                    kpi_counts: Dict[str, int] = {}
                    for a in anomalies:
                        kpi_counts[a.kpi_name] = kpi_counts.get(a.kpi_name, 0) + 1
                    top_kpi = max(kpi_counts.items(), key=lambda x: x[1])[0] if kpi_counts else None
                    evidence['anomaly_summary'] = {
                        'total_anomalies': len(anomalies),
                        'by_kpi': kpi_counts,
                        'top_kpi': top_kpi,
                        'filters': {
                            'site_id': site_id,
                            'date_range': [str(date_range[0]), str(date_range[1])] if date_range else None,
                        },
                    }
                else:
                    # Statistical fallback using z-score per KPI
                    df = data.copy()
                    # Apply filters
                    if site_id:
                        df = df[df['Site_ID'] == site_id]
                    if date_range and 'Date' in df.columns:
                        start, end = date_range
                        try:
                            df = df[(df['Date'] >= start) & (df['Date'] <= end)]
                        except Exception:
                            pass
                    # Determine KPIs present
                    kpis = [c for c in self.config.data.kpi_columns if c in df.columns]
                    by_kpi: Dict[str, int] = {}
                    by_site: Dict[str, int] = {}
                    for k in kpis:
                        series = df[k].dropna()
                        if series.empty:
                            continue
                        mu = series.mean()
                        sigma = series.std()
                        if sigma == 0 or np.isnan(sigma):
                            continue
                        mask = (df[k] - mu).abs() > 2.0 * sigma
                        an = df[mask & df[k].notna()].copy()
                        count_k = len(an)
                        by_kpi[k] = by_kpi.get(k, 0) + count_k
                        # Aggregate by site when no single site filter
                        if 'Site_ID' in an.columns:
                            if site_id is None:
                                for s, c in an['Site_ID'].value_counts().items():
                                    by_site[s] = by_site.get(s, 0) + int(c)
                            else:
                                by_site[site_id] = by_site.get(site_id, 0) + count_k
                    top_kpi = max(by_kpi.items(), key=lambda x: x[1])[0] if by_kpi else None
                    # Build fallback evidence
                    evidence['anomaly_summary'] = {
                        'total_anomalies': int(sum(by_kpi.values()) if by_kpi else 0),
                        'by_kpi': by_kpi,
                        'by_site': dict(sorted(by_site.items(), key=lambda x: x[1], reverse=True)[:10]) if by_site else {},
                        'top_kpi': top_kpi,
                        'method': 'zscore_fallback',
                        'filters': {
                            'site_id': site_id,
                            'date_range': [str(date_range[0]), str(date_range[1])] if date_range else None,
                        },
                    }
        
        except Exception as e:
            evidence['error'] = f"Error gathering evidence: {str(e)}"
        
        return evidence

    def _parse_query_filters(self, query: str, data) -> Tuple[Optional[str], Optional[Tuple[Any, Any]]]:
        """Extract site_id and a recent date_range (e.g., last week) from a natural language query."""
        import re
        site_id: Optional[str] = None
        date_range: Optional[Tuple[Any, Any]] = None
        try:
            # Site extraction: e.g., "site 2" or "site_id: 2"
            m = re.search(r"site\s*[:#-]?\s*(\w+)", query, re.IGNORECASE)
            if m:
                site_id = m.group(1)
            # Time window: last week / last N days
            days = None
            if re.search(r"last\s+week", query, re.IGNORECASE):
                days = 7
            else:
                m2 = re.search(r"last\s+(\d+)\s*days", query, re.IGNORECASE)
                if m2:
                    days = int(m2.group(1))
            if days is None:
                # Default to 7 days for anomaly summaries
                days = 7
            if 'Date' in data.columns:
                end_date = data['Date'].max()
                # Support pandas Timestamp and datetime
                try:
                    from datetime import timedelta
                    start_date = end_date - timedelta(days=days)
                except Exception:
                    start_date = end_date
                date_range = (start_date, end_date)
        except Exception:
            pass
        return site_id, date_range
    
    async def _synthesize_final_conclusion(self, query: str, reasoning_chain: List[ReasoningStep]) -> str:
        """Synthesize final conclusion from reasoning chain"""
        
        # Compile key points from each step
        step_summaries = []
        for step in reasoning_chain:
            summary = f"{step.reasoning_type.value}: {step.findings[0] if step.findings else 'No specific findings'}"
            step_summaries.append(summary)
        # If we have concrete anomaly evidence (top KPI), include it as factual context
        factual_block = ""
        anomaly_summary = None  # capture structured anomaly facts for post-processing
        try:
            for step in reasoning_chain:
                if step.evidence and 'anomaly_summary' in step.evidence:
                    a = step.evidence['anomaly_summary']
                    if a.get('by_kpi'):
                        factual_block = json.dumps({'anomaly_facts': a}, indent=2)
                        anomaly_summary = a
                        break
        except Exception:
            pass

        synthesis_prompt = f"""
Based on the comprehensive chain of thought analysis, provide a final conclusion for the original query.

ORIGINAL QUERY: {query}

REASONING CHAIN SUMMARY:
{chr(10).join(f"{i+1}. {summary}" for i, summary in enumerate(step_summaries))}

FACTUAL DATA (must be used if present):
{factual_block}

Please provide a concise but comprehensive final conclusion that:
1. Directly answers the original query
2. Integrates insights from all reasoning steps
3. Acknowledges any limitations or uncertainties
4. Provides clear, actionable insights

Keep the conclusion focused and professional.
"""
        
        try:
            generated = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._gemini_generate_text(
                    user_text=synthesis_prompt,
                    system_text="You summarize findings for a telecom analysis.",
                ),
            )
            # Post-process to guarantee explicit anomaly count mention
            if anomaly_summary and isinstance(generated, str):
                try:
                    total = anomaly_summary.get('total_anomalies') or anomaly_summary.get('total')
                    if total is not None:
                        lowered = generated.lower()
                        if 'detected anomalies' not in lowered and 'anomalies detected' not in lowered:
                            top_kpi = anomaly_summary.get('top_kpi')
                            method = anomaly_summary.get('method', anomaly_summary.get('source', 'model'))
                            extra = f"\nDetected anomalies: {total}"
                            if top_kpi:
                                extra += f" (top KPI: {top_kpi})"
                            if method:
                                extra += f" [source: {method}]"
                            generated = generated.rstrip() + "\n" + extra
                except Exception:
                    pass
            return generated
        except Exception as e:
            return f"Error synthesizing conclusion: {str(e)}"

    async def stream_final_conclusion(
        self,
        query: str,
        reasoning_chain: List[ReasoningStep],
        token_cb: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Stream final conclusion token-by-token if the LLM supports streaming.

        Falls back to a one-shot generation if streaming fails.
        """
        # Build the same prompt used by _synthesize_final_conclusion
        step_summaries = []
        for step in reasoning_chain:
            summary = f"{step.reasoning_type.value}: {step.findings[0] if step.findings else 'No specific findings'}"
            step_summaries.append(summary)

        synthesis_prompt = f"""
Based on the comprehensive chain of thought analysis, provide a final conclusion for the original query.

ORIGINAL QUERY: {query}

REASONING CHAIN SUMMARY:
{chr(10).join(f"{i+1}. {summary}" for i, summary in enumerate(step_summaries))}

Please provide a concise but comprehensive final conclusion that:
1. Directly answers the original query
2. Integrates insights from all reasoning steps
3. Acknowledges any limitations or uncertainties
4. Provides clear, actionable insights

Keep the conclusion focused and professional.
"""

        # If we already have a concise final answer (e.g., fast path), stream that as-is
        try:
            if not reasoning_chain:
                # Try to synthesize quickly from classify; but since we don't store, call non-streaming synth
                # However, the UI likely already has a final text; we stream a brief confirmation
                quick = await self._synthesize_final_conclusion(query, reasoning_chain)
                if token_cb and quick:
                    try:
                        token_cb(quick)
                    except Exception:
                        pass
                return quick

            import asyncio

            def _stream_sync() -> str:
                final_text_parts: list[str] = []
                # Build contents for Gemini client
                contents = [
                    {"role": "user", "parts": [synthesis_prompt]}
                ]
                try:
                    # Request streaming generation
                    for chunk in self._gm.generate_content(contents, stream=True):
                        text_part = getattr(chunk, "text", None)
                        if not text_part:
                            continue
                        final_text_parts.append(text_part)
                        if token_cb:
                            try:
                                token_cb(text_part)
                            except Exception:
                                # Ignore UI errors
                                pass
                    return "".join(final_text_parts)
                except Exception as e:
                    # Fallback to non-streaming
                    try:
                        full = self._gemini_generate_text(
                            user_text=synthesis_prompt,
                            system_text="You summarize findings for a telecom analysis.",
                        )
                        if token_cb and full:
                            try:
                                token_cb(full)
                            except Exception:
                                pass
                        return full
                    except Exception:
                        return f"Error synthesizing conclusion: {str(e)}"

            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _stream_sync)
        except Exception as e:
            # As a last resort, call non-streaming synthesizer
            return await self._synthesize_final_conclusion(query, reasoning_chain)
    
    def _extract_key_insights(self, reasoning_chain: List[ReasoningStep]) -> List[str]:
        """Extract key insights from the reasoning chain"""
        insights = []
        
        for step in reasoning_chain:
            # Add high-confidence findings as insights
            if step.confidence > 0.7 and step.findings:
                insights.extend(step.findings[:2])  # Top 2 findings per step
        
        # Remove duplicates and filter
        unique_insights = []
        for insight in insights:
            if len(insight) > 20 and insight not in unique_insights:
                unique_insights.append(insight)
        
        return unique_insights[:10]  # Limit to top 10 insights
    
    def _extract_recommendations(self, reasoning_chain: List[ReasoningStep]) -> List[str]:
        """Extract recommendations from the reasoning chain"""
        recommendations = []
        
        # Look for recommendation step specifically
        for step in reasoning_chain:
            if step.reasoning_type == ReasoningType.RECOMMENDATION_GENERATION:
                recommendations.extend(step.findings)
        
        # If no specific recommendations, extract actionable items from other steps
        if not recommendations:
            action_words = ['should', 'recommend', 'suggest', 'need to', 'must', 'consider']
            for step in reasoning_chain:
                for finding in step.findings:
                    if any(word in finding.lower() for word in action_words):
                        recommendations.append(finding)
        
        return recommendations[:5]  # Limit to top 5 recommendations
    
    def _calculate_overall_confidence(self, reasoning_chain: List[ReasoningStep]) -> float:
        """Calculate overall confidence based on individual step confidences"""
        if not reasoning_chain:
            return 0.0
        
        confidences = [step.confidence for step in reasoning_chain if step.confidence > 0]
        if not confidences:
            return 0.5
        
        # Use weighted average with more weight on later steps
        weights = [i + 1 for i in range(len(confidences))]
        weighted_sum = sum(conf * weight for conf, weight in zip(confidences, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.5
    
    async def explain_reasoning(self, result: ChainOfThoughtResult) -> str:
        """Generate a human-readable explanation of the reasoning process"""
        
        explanation = f"""
CHAIN OF THOUGHT ANALYSIS EXPLANATION

Query: {result.query}

Reasoning Process:
"""
        
        for i, step in enumerate(result.reasoning_chain, 1):
            explanation += f"""
Step {i}: {step.reasoning_type.value.replace('_', ' ').title()}
- Objective: {step.question}
- Key Findings: {', '.join(step.findings[:3])}
- Confidence: {step.confidence:.1%}
"""
        
        explanation += f"""
Final Conclusion:
{result.final_conclusion}

Key Insights:
{chr(10).join(f"• {insight}" for insight in result.key_insights)}

Recommendations:
{chr(10).join(f"• {rec}" for rec in result.recommendations)}

Overall Confidence: {result.confidence:.1%}
Processing Time: {result.total_processing_time_ms:.0f}ms
"""
        
        return explanation


def create_chain_of_thought_agent(config: TelecomConfig) -> ChainOfThoughtAgent:
    """
    Factory function to create a configured chain of thought agent
    
    Args:
        config: Configuration object
        
    Returns:
        Configured ChainOfThoughtAgent instance
    """
    return ChainOfThoughtAgent(config)
