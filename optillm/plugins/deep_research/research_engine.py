"""
Deep Research Engine - Core Implementation

This module implements the Test-Time Diffusion Deep Researcher (TTD-DR) algorithm
as described in "Deep Researcher with Test-Time Diffusion" (https://arxiv.org/abs/2507.16075v1).

The TTD-DR approach treats research as a diffusion process with iterative refinement
through denoising and retrieval, generating comprehensive research reports.
"""

import re
from typing import Tuple, List, Dict, Any
from datetime import datetime
from optillm.plugins.web_search_plugin import run as web_search_run
from optillm.plugins.readurls_plugin import run as readurls_run
from optillm.plugins.deep_research.session_state import (
    get_session_manager,
    close_session,
)
import uuid


def clean_reasoning_tags(text: str) -> str:
    """
    Remove reasoning tags from model responses for clean final output.

    Removes common reasoning tags like:
    - <think></think>
    - <thinking></thinking>
    - <reasoning></reasoning>
    - <thought></thought>

    Args:
        text: Raw model response text

    Returns:
        Cleaned text with reasoning tags removed
    """
    if not text:
        return text

    # List of reasoning tag patterns to remove
    reasoning_patterns = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
        r"<thought>.*?</thought>",
        r"<reflect>.*?</reflect>",
        r"<reflection>.*?</reflection>",
    ]

    cleaned_text = text
    for pattern in reasoning_patterns:
        # Use DOTALL flag to match across newlines
        cleaned_text = re.sub(
            pattern, "", cleaned_text, flags=re.DOTALL | re.IGNORECASE
        )

    # Clean up any extra whitespace left behind, but preserve markdown formatting
    cleaned_text = re.sub(
        r"\n\s*\n\s*\n+", "\n\n", cleaned_text
    )  # Multiple empty lines to double
    cleaned_text = re.sub(
        r"  +", " ", cleaned_text
    )  # Multiple spaces to single space (but preserve intentional double spaces)
    cleaned_text = cleaned_text.strip()

    return cleaned_text


def cleanup_placeholder_tags(text: str) -> str:
    """
    Remove any remaining placeholder tags from the final report.

    This is a final cleanup step to ensure no incomplete research tags remain
    in the published report.

    Args:
        text: Research report text

    Returns:
        Text with all placeholder tags removed
    """
    if not text:
        return text

    # Comprehensive patterns for research placeholder tags
    placeholder_patterns = [
        # Research placeholders
        r"\[NEEDS RESEARCH[^\]]*\]",
        r"\[SOURCE NEEDED[^\]]*\]",
        r"\[RESEARCH NEEDED[^\]]*\]",
        r"\[CITATION NEEDED[^\]]*\]",
        r"\[MORE RESEARCH NEEDED[^\]]*\]",
        r"\[REQUIRES INVESTIGATION[^\]]*\]",
        r"\[TO BE RESEARCHED[^\]]*\]",
        r"\[VERIFY[^\]]*\]",
        r"\[CHECK[^\]]*\]",
        # Citation placeholders (like your example)
        r"\[Placeholder for[^\]]+\]",
        r"\[\d+\]\s*\[Placeholder[^\]]+\]",
        r"\[Insert citation[^\]]*\]",
        r"\[Add reference[^\]]*\]",
        r"\[Reference needed[^\]]*\]",
        # Content placeholders
        r"\[To be completed[^\]]*\]",
        r"\[Under development[^\]]*\]",
        r"\[Coming soon[^\]]*\]",
        r"\[TBD[^\]]*\]",
        r"\[TODO[^\]]*\]",
        # Question placeholders and incomplete sections
        r"\[Question \d+[^\]]*\]",
        r"\[Research question[^\]]*\]",
    ]

    cleaned_text = text
    for pattern in placeholder_patterns:
        # Remove the placeholder tags
        cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    # Also remove any sentences that are entirely placeholder-based
    lines = cleaned_text.split("\n")
    filtered_lines = []

    for line in lines:
        # Skip lines that are mostly just removed placeholders (now empty or just punctuation)
        stripped = line.strip()
        if stripped and not re.match(r"^[\s\-\*\.\,\;\:]*$", stripped):
            filtered_lines.append(line)
        elif not stripped:  # Keep empty lines for formatting
            filtered_lines.append(line)

    # Rejoin and clean up extra whitespace
    result = "\n".join(filtered_lines)
    result = re.sub(r"\n\s*\n\s*\n+", "\n\n", result)  # Multiple empty lines to double
    result = result.strip()

    return result


def validate_citation_usage(text: str, total_citations: int) -> Dict[str, Any]:
    """
    Validate that citations are actually used in the report text.

    Checks which citation numbers appear in the body text and warns about
    unused citations that are only in the references section.

    Args:
        text: The full report text including references
        total_citations: Total number of citations available

    Returns:
        Dict with validation results including used/unused citation counts
    """
    if not text:
        return {
            "citations_used": 0,
            "citations_total": total_citations,
            "usage_percentage": 0.0,
            "unused_citations": list(range(1, total_citations + 1)),
            "warning": "Empty report text",
        }

    # Find all citation references in the body text (before References section)
    body_text = text.split("## References")[0] if "## References" in text else text

    # Extract all citation numbers from body text using regex
    citations_in_body = set()
    citation_pattern = r"\[(\d+)\]"
    for match in re.finditer(citation_pattern, body_text):
        citation_num = int(match.group(1))
        citations_in_body.add(citation_num)

    # Calculate unused citations
    all_citations = set(range(1, total_citations + 1))
    unused_citations = sorted(all_citations - citations_in_body)

    usage_percentage = (
        (len(citations_in_body) / total_citations * 100) if total_citations > 0 else 0
    )

    result = {
        "citations_used": len(citations_in_body),
        "citations_total": total_citations,
        "usage_percentage": usage_percentage,
        "unused_citations": unused_citations,
        "used_citations": sorted(citations_in_body),
    }

    # Add warnings if citation usage is low
    if usage_percentage < 30:
        result["warning"] = (
            f"Very low citation usage ({usage_percentage:.1f}%). Most sources are not cited in the text."
        )
    elif usage_percentage < 50:
        result["warning"] = (
            f"Low citation usage ({usage_percentage:.1f}%). Many sources are not cited in the text."
        )

    return result


def validate_report_completeness(text: str) -> Dict[str, Any]:
    """
    Validate that the research report is complete and ready for publication.

    Checks for:
    - Placeholder citations
    - Incomplete sections
    - Unfinished research questions
    - Missing content indicators

    Returns:
        Dict with validation results and suggestions for fixes
    """
    if not text:
        return {"is_complete": False, "issues": ["Empty report"], "suggestions": []}

    issues = []
    suggestions = []

    # Check for placeholder citations
    placeholder_citation_patterns = [
        r"\[Placeholder for[^\]]+\]",
        r"\[\d+\]\s*\[Placeholder[^\]]+\]",
        r"\[Insert citation[^\]]*\]",
        r"\[Reference needed[^\]]*\]",
    ]

    for pattern in placeholder_citation_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            issues.append(f"Found {len(matches)} placeholder citations: {matches[:3]}")
            suggestions.append(
                "Replace placeholder citations with actual sources or remove incomplete claims"
            )

    # Check for incomplete research questions sections
    if "Research Questions for Investigation" in text:
        # Look for sections that seem to be lists of questions without answers
        question_section_match = re.search(
            r"## Research Questions for Investigation.*?(?=##|$)", text, re.DOTALL
        )
        if question_section_match:
            question_content = question_section_match.group(0)
            # Count questions vs answers
            question_lines = [
                line
                for line in question_content.split("\n")
                if line.strip().startswith("*") or line.strip().startswith("-")
            ]
            if len(question_lines) > 3:  # Many unanswered questions
                issues.append("Report contains unanswered research questions section")
                suggestions.append(
                    "Convert research questions into answered findings or remove incomplete section"
                )

    # Check for incomplete sections (sections with only placeholders)
    section_pattern = r"##\s+([^#\n]+)\n(.*?)(?=##|$)"
    sections = re.findall(section_pattern, text, re.DOTALL)

    for section_title, section_content in sections:
        # Check if section is mostly placeholders
        placeholder_count = len(
            re.findall(
                r"\[[^\]]*(?:placeholder|needed|research|todo|tbd)[^\]]*\]",
                section_content,
                re.IGNORECASE,
            )
        )
        content_lines = [
            line.strip() for line in section_content.split("\n") if line.strip()
        ]

        if placeholder_count > len(content_lines) / 3:  # More than 1/3 placeholders
            issues.append(f"Section '{section_title.strip()}' is mostly placeholders")
            suggestions.append(
                f"Complete content for '{section_title.strip()}' section or remove it"
            )

    # Check for incomplete reference lists
    if text.count("[") - text.count("]") != 0:
        issues.append("Unmatched brackets detected - possible incomplete citations")
        suggestions.append("Review and fix citation formatting")

    # Check for very short sections that might be incomplete
    if len(text.split()) < 500:  # Very short report
        issues.append("Report appears to be very short, possibly incomplete")
        suggestions.append("Ensure all research areas are adequately covered")

    is_complete = len(issues) == 0

    return {
        "is_complete": is_complete,
        "issues": issues,
        "suggestions": suggestions,
        "word_count": len(text.split()),
        "section_count": len(sections),
    }


class DeepResearcher:
    """
    Simplified implementation of Test-Time Diffusion Deep Researcher (TTD-DR) algorithm

    This class implements the core concepts from the TTD-DR paper: treating research as a
    diffusion process with iterative refinement through denoising and retrieval.

    Implemented features:
    - Preliminary draft generation (updatable skeleton)
    - Gap analysis and draft-guided search
    - Iterative denoising through retrieval
    - Quality-guided termination

    Not yet implemented (future work):
    - Component-wise self-evolutionary optimization (fitness tracking exists but not used)
    - Memory-based synthesis for unbounded context

    Based on: https://arxiv.org/abs/2507.16075v1
    """

    def __init__(
        self, client, model: str, max_iterations: int = 5, max_sources: int = 30
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.max_sources = max_sources
        self.session_id = str(uuid.uuid4())  # Unique session ID for this research
        self.session_manager = None  # Will be set when research starts
        self.research_state = {"iteration": 0}  # Track current iteration for metadata
        self.total_tokens = 0
        self.citations = {}  # Map citation number to source info
        self.citation_counter = 0
        self.source_content_map = {}  # Map URL to content for citations

        # TTD-DR specific components
        self.current_draft = ""  # Persistent evolving draft
        self.draft_history = []  # Track draft evolution
        self.component_fitness = {  # Self-evolution fitness tracking
            "search_strategy": 1.0,
            "synthesis_quality": 1.0,
            "gap_detection": 1.0,
            "integration_ability": 1.0,
        }
        self.gap_analysis_history = []  # Track identified gaps over time
        self.session_manager = None  # Browser session manager for web searches

    def cleanup_placeholder_tags(self, text: str) -> str:
        """
        Remove any remaining placeholder tags from the final report.

        This is a final cleanup step to ensure no incomplete research tags remain
        in the published report.

        Args:
            text: Research report text

        Returns:
            Text with all placeholder tags removed
        """
        return cleanup_placeholder_tags(text)

    def fix_incomplete_report(
        self, report: str, validation: Dict[str, Any], original_query: str
    ) -> str:
        """
        Attempt to fix an incomplete report by removing problematic sections
        and ensuring a coherent final document.

        This is a fallback when the report contains placeholders or incomplete sections.
        """
        print("🔧 Attempting to fix incomplete report...")

        # Start with the basic cleanup
        fixed_report = cleanup_placeholder_tags(report)

        # Remove sections that are mostly placeholders or incomplete
        if "Research Questions for Investigation" in fixed_report:
            # Remove unanswered research questions sections
            fixed_report = re.sub(
                r"## Research Questions for Investigation.*?(?=##|$)",
                "",
                fixed_report,
                flags=re.DOTALL,
            )
            print("   - Removed incomplete research questions section")

        # Remove citation placeholders from reference section
        fixed_report = re.sub(r"\[\d+\]\s*\[Placeholder[^\]]+\]\n?", "", fixed_report)

        # Clean up any empty sections
        fixed_report = re.sub(r"##\s+([^#\n]+)\n\s*(?=##)", "", fixed_report)

        # If report is still very short, add a completion note
        if len(fixed_report.split()) < 300:
            completion_note = f"""
            
## Note on Report Completion

This research report represents the findings gathered during the available research time. While comprehensive coverage was the goal, some areas may require additional investigation for complete analysis.

For more detailed information on specific aspects of {original_query}, additional focused research sessions may be beneficial.
"""
            # Insert before references section if it exists
            if "## References" in fixed_report:
                fixed_report = fixed_report.replace(
                    "## References", completion_note + "\n## References"
                )
            else:
                fixed_report += completion_note

            print("   - Added completion note due to short report length")

        # Final cleanup
        fixed_report = re.sub(r"\n\s*\n\s*\n+", "\n\n", fixed_report)
        fixed_report = fixed_report.strip()

        # Validate the fix
        new_validation = validate_report_completeness(fixed_report)
        if new_validation["is_complete"]:
            print("✅ Report successfully fixed and validated")
        else:
            print(
                f"⚠️  Report still has {len(new_validation['issues'])} issues after fixing"
            )

        return fixed_report

    def decompose_query(self, system_prompt: str, initial_query: str) -> List[str]:
        """
        Decompose complex research query into focused sub-queries
        This implements the query planning phase of TTD-DR
        """
        decomposition_prompt = f"""
        You are a research assistant. Given a complex query, break it down into 3-5 focused sub-queries that would help gather comprehensive information.
        
        Original query: {initial_query}
        
        Provide sub-queries in this format:
        1. [specific focused question]
        2. [specific focused question]
        3. [specific focused question]
        ...
        
        Make each sub-query specific and searchable. Focus on different aspects of the main topic.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": decomposition_prompt},
                ],
                temperature=0.7,
                max_tokens=1000,
            )

            if not response.choices:
                raise ValueError("LLM returned empty response in decompose_query")
            content = response.choices[0].message.content.strip()
            # Clean reasoning tags from query decomposition response
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens

            # Extract numbered queries
            queries = []
            for line in content.split("\n"):
                line = line.strip()
                if re.match(r"^\d+\.", line):
                    query = re.sub(r"^\d+\.\s*\[?(.*?)\]?$", r"\1", line).strip()
                    if query:
                        queries.append(query)

            return queries[:5]  # Limit to 5 sub-queries

        except Exception:
            # Fallback: use original query
            return [initial_query]

    def perform_web_search(self, queries: List[str]) -> str:
        """
        Perform web search for multiple queries using the web_search plugin
        """
        all_results = []

        # Check if session manager is available
        if not hasattr(self, "session_manager") or self.session_manager is None:
            # Log warning - this shouldn't happen in normal flow
            print(
                f"⚠️  Warning: session_manager not available in perform_web_search (session_id: {getattr(self, 'session_id', 'N/A')})"
            )
            self.session_manager = None
        else:
            print(
                f"📊 Using existing session manager for web search (session_id: {self.session_id}, manager: {id(self.session_manager)})"
            )

        # Perform individual searches for each query to avoid truncation issues
        for i, query in enumerate(queries):
            try:
                # Format as a clean search query
                search_query = f"search for {query.strip()}"

                # Perform search with reduced results per query to stay within limits
                results_per_query = max(1, self.max_sources // len(queries))

                enhanced_query, _ = web_search_run(
                    "",
                    search_query,
                    None,
                    None,
                    {
                        "num_results": results_per_query,
                        "delay_seconds": None,  # Use default random delay (4-32 seconds)
                        "headless": False,  # Allow CAPTCHA solving if needed
                        "session_manager": self.session_manager,  # Use shared browser session
                    },
                )

                if enhanced_query and "Web Search Results" in enhanced_query:
                    all_results.append(enhanced_query)

            except Exception as e:
                # Continue with other queries if one fails
                all_results.append(f"Search failed for query '{query}': {str(e)}")
                continue

        if not all_results:
            return "Web search failed: No results obtained from any query"

        # Combine all search results
        combined_results = "\n\n".join(all_results)
        return combined_results

    def extract_and_fetch_urls(self, search_results: str) -> Tuple[str, List[Dict]]:
        """
        Extract URLs from search results and fetch their content using readurls plugin
        Returns content and list of sources with metadata
        """
        try:
            # First extract URLs and metadata from search results
            sources = []

            # Pattern to match search result blocks
            result_pattern = r"(\d+)\.\s*\*\*(.+?)\*\*\s*\n\s*URL:\s*(.+?)\n"
            matches = re.findall(result_pattern, search_results, re.MULTILINE)

            for match in matches:
                source = {
                    "number": match[0],
                    "title": match[1].strip(),
                    "url": match[2].strip(),
                    "access_date": datetime.now().strftime("%Y-%m-%d"),
                }
                sources.append(source)

            # If regex doesn't work, try line-by-line parsing
            if not sources:
                lines = search_results.split("\n")
                current_source = {}

                for i, line in enumerate(lines):
                    # Check for numbered item with title
                    title_match = re.match(r"^(\d+)\.\s*\*\*(.+?)\*\*", line.strip())
                    if title_match:
                        if current_source and "url" in current_source:
                            sources.append(current_source)
                        current_source = {
                            "number": title_match.group(1),
                            "title": title_match.group(2).strip(),
                        }
                    # Check for URL line
                    elif line.strip().startswith("URL:") and current_source:
                        url = line.strip()[4:].strip()
                        current_source["url"] = url
                        current_source["access_date"] = datetime.now().strftime(
                            "%Y-%m-%d"
                        )

                if current_source and "url" in current_source:
                    sources.append(current_source)

            # Fetch content for all URLs
            content_with_urls, _ = readurls_run("", search_results, None, None)

            return content_with_urls, sources
        except Exception as e:
            return f"URL fetching failed: {str(e)}", []

    def evaluate_completeness(
        self, system_prompt: str, query: str, current_synthesis: str
    ) -> Tuple[bool, List[str]]:
        """
        Evaluate if the current research is complete or needs more information
        Returns (is_complete, list_of_missing_aspects)
        """
        evaluation_prompt = f"""
        You are evaluating the completeness of a research synthesis. 
        
        Original query: {query}
        Current synthesis: {current_synthesis}
        
        Evaluate if this synthesis adequately addresses the original query. Consider:
        1. Are all major aspects of the query covered?
        2. Is there sufficient depth and detail?
        3. Are there any obvious gaps or missing information?
        
        Respond in this format:
        COMPLETE: [YES/NO]
        MISSING: [list any missing aspects, one per line, or "None" if complete]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.3,
                max_tokens=500,
            )

            if not response.choices:
                raise ValueError("LLM returned empty response in evaluate_completeness")
            content = response.choices[0].message.content.strip()
            # Clean reasoning tags from completeness evaluation response
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens

            # Parse response
            is_complete = "COMPLETE: YES" in content.upper()

            missing_aspects = []
            if "MISSING:" in content.upper():
                missing_section = content.split("MISSING:")[-1].strip()
                if missing_section.upper() != "NONE":
                    missing_aspects = [
                        line.strip()
                        for line in missing_section.split("\n")
                        if line.strip()
                    ]

            return is_complete, missing_aspects

        except Exception:
            # Default to not complete on error
            return False, ["Error in evaluation"]

    def generate_focused_queries(
        self, missing_aspects: List[str], original_query: str
    ) -> List[str]:
        """
        Generate focused search queries to address missing aspects
        """
        focused_queries = []
        for aspect in missing_aspects:
            # Create a focused query combining the original topic with the missing aspect
            focused_query = f"{original_query} {aspect}"
            focused_queries.append(focused_query)

        return focused_queries[:3]  # Limit to 3 additional queries per iteration

    def generate_preliminary_draft(self, system_prompt: str, initial_query: str) -> str:
        """
        Generate the preliminary draft (updatable skeleton) from LLM internal knowledge
        This serves as the initial state for the diffusion process
        """
        draft_prompt = f"""
        Generate a preliminary research report structure for the following query using your internal knowledge.
        This will serve as an evolving draft that gets refined through iterative research.
        
        Query: {initial_query}
        
        Create a structured report with:
        1. Title and Executive Summary (brief)
        2. Introduction and Background (what you know)
        3. Key Areas to Explore (identify knowledge gaps)
        4. Preliminary Findings (from internal knowledge)
        5. Research Questions for Investigation
        6. Conclusion (preliminary thoughts)
        
        IMPORTANT: You MUST mark multiple areas that need external research with [NEEDS RESEARCH] tags.
        Every claim that would benefit from external evidence should have [SOURCE NEEDED].
        This is a preliminary draft - it should have many gaps for iterative improvement.
        
        Example of proper marking:
        - "Recent studies show [SOURCE NEEDED] that quantum computing..."
        - "The economic impact [NEEDS RESEARCH: current market data] is significant..."
        - "Historical context [NEEDS RESEARCH: specific timeline and events] shows..."
        
        Include AT LEAST 5-10 [NEEDS RESEARCH] or [SOURCE NEEDED] tags throughout the draft.
        Be explicit about what you don't know and what needs external validation.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": draft_prompt},
                ],
                temperature=0.7,
                max_tokens=2000,
            )

            if not response.choices:
                raise ValueError(
                    "LLM returned empty response in generate_preliminary_draft"
                )
            draft = response.choices[0].message.content.strip()
            draft = clean_reasoning_tags(draft)
            self.total_tokens += response.usage.completion_tokens

            return draft

        except Exception as e:
            return f"Failed to generate preliminary draft: {str(e)}"

    def analyze_draft_gaps(
        self, current_draft: str, original_query: str
    ) -> List[Dict[str, str]]:
        """
        Analyze the current draft to identify gaps, weaknesses, and areas needing research
        This guides the next retrieval iteration (draft-guided search)
        """
        gap_analysis_prompt = f"""
        Analyze the following research draft to identify specific gaps and areas that need external research.
        Be thorough and aggressive in finding areas for improvement - even good drafts can be enhanced.
        
        Original Query: {original_query}
        
        Current Draft:
        {current_draft}
        
        CRITICAL ANALYSIS REQUIRED:
        1. MANDATORY: Find ALL [NEEDS RESEARCH], [SOURCE NEEDED], [CITATION NEEDED] tags
        2. Identify claims lacking evidence (even if not explicitly marked)
        3. Find areas that could benefit from recent data or statistics
        4. Spot generalizations that need specific examples
        5. Locate outdated information or areas needing current updates
        6. Identify missing perspectives or counterarguments
        
        For each gap you identify, provide:
        1. SECTION: Which section has the gap
        2. GAP_TYPE: [PLACEHOLDER_TAG, MISSING_INFO, OUTDATED_INFO, NEEDS_EVIDENCE, LACKS_DEPTH, NEEDS_EXAMPLES, MISSING_PERSPECTIVE]
        3. SPECIFIC_NEED: Exactly what information is needed
        4. SEARCH_QUERY: A specific, targeted search query to address this gap
        5. PRIORITY: [HIGH, MEDIUM, LOW] - HIGH for placeholder tags and critical missing info
        
        Format each gap as:
        GAP_ID: [number]
        SECTION: [section name]
        GAP_TYPE: [type]
        SPECIFIC_NEED: [what's missing]
        SEARCH_QUERY: [search query to find this info]
        PRIORITY: [priority level]
        
        IMPORTANT: Identify AT LEAST 3-8 gaps. Be critical and thorough.
        Even well-written sections can benefit from additional evidence, examples, or perspectives.
        Push for depth, accuracy, and comprehensiveness in the research.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research analyst.",
                    },
                    {"role": "user", "content": gap_analysis_prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            if not response.choices:
                raise ValueError("LLM returned empty response in analyze_draft_gaps")
            content = response.choices[0].message.content.strip()
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens

            # Parse the gaps
            gaps = []
            current_gap = {}

            for line in content.split("\n"):
                line = line.strip()
                if line.startswith("GAP_ID:"):
                    if current_gap:
                        gaps.append(current_gap)
                    current_gap = {"id": line.split(":", 1)[1].strip()}
                elif line.startswith("SECTION:"):
                    current_gap["section"] = line.split(":", 1)[1].strip()
                elif line.startswith("GAP_TYPE:"):
                    current_gap["gap_type"] = line.split(":", 1)[1].strip()
                elif line.startswith("SPECIFIC_NEED:"):
                    current_gap["specific_need"] = line.split(":", 1)[1].strip()
                elif line.startswith("SEARCH_QUERY:"):
                    current_gap["search_query"] = line.split(":", 1)[1].strip()
                elif line.startswith("PRIORITY:"):
                    current_gap["priority"] = line.split(":", 1)[1].strip()

            if current_gap:
                gaps.append(current_gap)

            return gaps

        except Exception:
            # Fallback: create basic gaps from the draft
            return [
                {
                    "id": "1",
                    "section": "General",
                    "gap_type": "MISSING_INFO",
                    "specific_need": "More detailed information needed",
                    "search_query": original_query,
                }
            ]

    def perform_gap_targeted_search(self, gaps: List[Dict[str, str]]) -> str:
        """
        Perform targeted searches based on identified gaps in the current draft
        Prioritizes HIGH priority gaps (placeholder tags) first
        """
        all_results = []

        # Check if session manager is available
        if not hasattr(self, "session_manager") or self.session_manager is None:
            # Log warning - this shouldn't happen in normal flow
            print("⚠️  Warning: session_manager not available in perform_web_search")
            self.session_manager = None

        # Sort gaps by priority - HIGH priority first (placeholder tags)
        sorted_gaps = sorted(
            gaps,
            key=lambda g: (
                0
                if g.get("priority", "").upper() == "HIGH"
                else 1 if g.get("priority", "").upper() == "MEDIUM" else 2
            ),
        )

        for gap in sorted_gaps:
            search_query = gap.get("search_query", "")
            if not search_query:
                continue

            try:
                # Format as a clean search query
                search_query = f"search for {search_query.strip()}"

                # Perform search with context about what gap we're filling
                enhanced_query, _ = web_search_run(
                    "",
                    search_query,
                    None,
                    None,
                    {
                        "num_results": max(1, self.max_sources // len(gaps)),
                        "delay_seconds": None,  # Use default random delay (4-32 seconds)
                        "headless": False,
                        "session_manager": self.session_manager,  # Use shared browser session
                    },
                )

                if enhanced_query and "Web Search Results" in enhanced_query:
                    # Tag results with gap context
                    gap_context = f"[ADDRESSING GAP: {gap.get('section', 'Unknown')} - {gap.get('specific_need', 'General research')}]\n"
                    all_results.append(gap_context + enhanced_query)

            except Exception:
                continue

        return (
            "\n\n".join(all_results)
            if all_results
            else "No gap-targeted search results obtained"
        )

    def denoise_draft_with_retrieval(
        self, current_draft: str, retrieval_content: str, original_query: str
    ) -> str:
        """
        Core denoising step: integrate retrieved information with current draft
        This is the heart of the diffusion process
        """
        # Build citation context from currently available sources
        citation_context = "\n\n**AVAILABLE CITATIONS (USE THESE!):**\n"
        for num, source in self.citations.items():
            citation_context += f"[{num}] {source.get('title', 'Untitled')} - {source.get('url', 'No URL')}\n"

        denoising_prompt = f"""
        You are performing a denoising step in a research diffusion process.

        TASK: Integrate new retrieved information with the existing draft to reduce "noise" (gaps, inaccuracies, incompleteness).

        Original Query: {original_query}

        Current Draft:
        {current_draft}

        New Retrieved Information:
        {retrieval_content}
        {citation_context}

        CRITICAL CITATION REQUIREMENTS:
        1. EVERY factual claim, statistic, finding, or piece of evidence MUST be cited using [1], [2], etc.
        2. Multiple related claims from the same source can share a citation [3]
        3. Claims from different sources should have multiple citations like [1,4,7]
        4. Direct quotes MUST have citations immediately after the closing quote
        5. When integrating new information, ALWAYS add appropriate citations from the available sources
        6. Uncited claims will be considered incomplete and must be fixed

        DENOISING INSTRUCTIONS:
        1. Identify where the new information fills gaps marked with [NEEDS RESEARCH] or [SOURCE NEEDED]
        2. Replace placeholder content with specific, detailed information from the retrieved content
        3. Add proper citations for ALL new information using the citation numbers shown above
        4. Resolve any conflicts between new and existing information
        5. Maintain the overall structure and coherence of the draft
        6. Enhance depth and accuracy without losing existing valuable insights
        7. Mark any remaining research needs with [NEEDS RESEARCH]
        8. Review the draft and add missing citations to any uncited factual claims

        QUALITY CHECK: Before returning, verify that the majority of substantive claims have citations.
        Aim for at least 70% of factual statements to be properly cited.

        Return the improved draft with integrated information and comprehensive citations.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research synthesizer performing draft denoising.",
                    },
                    {"role": "user", "content": denoising_prompt},
                ],
                temperature=0.6,
                max_tokens=3000,
            )

            if not response.choices:
                raise ValueError(
                    "LLM returned empty response in denoise_draft_with_retrieval"
                )
            denoised_draft = response.choices[0].message.content.strip()
            denoised_draft = clean_reasoning_tags(denoised_draft)
            self.total_tokens += response.usage.completion_tokens

            return denoised_draft

        except Exception as e:
            return f"Denoising failed: {str(e)}\n\nFalling back to current draft:\n{current_draft}"

    def evaluate_draft_quality(
        self, draft: str, previous_draft: str, original_query: str
    ) -> Dict[str, float]:
        """
        Evaluate the quality improvement of the current draft vs previous iteration
        Used for termination decisions and component fitness updates
        """
        evaluation_prompt = f"""
        Evaluate the research draft quality improvement.
        
        Original Query: {original_query}
        
        Previous Draft:
        {previous_draft}
        
        Current Draft:
        {draft}
        
        Rate the following aspects from 0.0 to 1.0:
        
        COMPLETENESS: How well does the current draft address all aspects of the query?
        ACCURACY: How accurate and reliable is the information?
        DEPTH: How detailed and comprehensive is the analysis?
        COHERENCE: How well-structured and logically organized is the draft?
        CITATIONS: How well are sources cited and integrated?
        IMPROVEMENT: How much better is this draft compared to the previous version?
        
        Respond ONLY with:
        COMPLETENESS: [score]
        ACCURACY: [score]
        DEPTH: [score]
        COHERENCE: [score]
        CITATIONS: [score]
        IMPROVEMENT: [score]
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert research quality evaluator.",
                    },
                    {"role": "user", "content": evaluation_prompt},
                ],
                temperature=0.2,
                max_tokens=500,
            )

            if not response.choices:
                raise ValueError(
                    "LLM returned empty response in evaluate_draft_quality"
                )
            content = response.choices[0].message.content.strip()
            content = clean_reasoning_tags(content)
            self.total_tokens += response.usage.completion_tokens

            # Parse scores
            scores = {}
            for line in content.split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().lower()
                    try:
                        scores[key] = float(value.strip())
                    except ValueError:
                        scores[key] = 0.5  # Default score

            return scores

        except Exception:
            # Default scores
            return {
                "completeness": 0.5,
                "accuracy": 0.5,
                "depth": 0.5,
                "coherence": 0.5,
                "citations": 0.5,
                "improvement": 0.1,
            }

    def update_component_fitness(self, quality_scores: Dict[str, float]):
        """
        Update component fitness based on performance (self-evolution)

        NOTE: This method tracks fitness values but they are not yet used to modify
        component behavior. This is placeholder code for future implementation of
        component-wise self-evolutionary optimization as described in the TTD-DR paper.
        """
        # Update fitness based on quality improvements
        improvement = quality_scores.get("improvement", 0.0)

        if improvement > 0.1:  # Significant improvement
            self.component_fitness["search_strategy"] *= 1.1
            self.component_fitness["synthesis_quality"] *= 1.1
            self.component_fitness["integration_ability"] *= 1.1
        elif improvement < 0.05:  # Poor improvement
            self.component_fitness["search_strategy"] *= 0.95
            self.component_fitness["synthesis_quality"] *= 0.95

        # Cap fitness values
        for key in self.component_fitness:
            self.component_fitness[key] = max(
                0.1, min(2.0, self.component_fitness[key])
            )

    def research(self, system_prompt: str, initial_query: str) -> Tuple[str, int]:
        """
        TTD-DR (Test-Time Diffusion Deep Researcher) main algorithm

        Implements the core diffusion process with:
        1. Preliminary draft generation (initial noisy state)
        2. Initial research to gather external sources
        3. Iterative denoising through draft-guided retrieval
        4. Gap analysis to identify areas needing more research
        5. Quality-guided termination

        Note: Component-wise self-evolutionary optimization is tracked but not yet
        used to modify behavior (future enhancement).
        """

        # Get or create a browser session for this research session
        self.session_manager = get_session_manager(
            self.session_id, headless=False, timeout=30
        )
        if self.session_manager:
            print(
                f"🔬 Starting deep research with session ID: {self.session_id} (DeepResearcher instance: {id(self)})"
            )
        else:
            print("⚠️ Failed to create browser session, proceeding without web search")

        try:
            # PHASE 1: INITIALIZATION - Generate preliminary draft (updatable skeleton)
            print("TTD-DR: Generating preliminary draft...")
            self.current_draft = self.generate_preliminary_draft(
                system_prompt, initial_query
            )
            self.draft_history.append(self.current_draft)

            # PHASE 1.5: INITIAL RESEARCH - Ensure we always gather external sources
            print("TTD-DR: Performing initial research...")
            initial_queries = self.decompose_query(system_prompt, initial_query)
            if initial_queries:
                print(f"  - Searching for {len(initial_queries)} initial topics...")
                initial_search_results = self.perform_web_search(initial_queries)

                # Extract and fetch URLs from initial search
                if (
                    initial_search_results
                    and "Web Search Results" in initial_search_results
                ):
                    print("  - Extracting initial sources...")
                    initial_content, initial_sources = self.extract_and_fetch_urls(
                        initial_search_results
                    )

                    # Register initial sources for citations
                    for source in initial_sources:
                        if "url" in source:
                            self.citation_counter += 1
                            self.citations[self.citation_counter] = source

                    print(f"  - Found {len(initial_sources)} initial sources")
                else:
                    print("  - No sources found in initial search")
            else:
                print("  - Warning: Could not decompose query for initial research")
                # Fallback: Create simple search queries from the original query
                print("  - Using fallback search strategy...")
                fallback_queries = [
                    initial_query
                ]  # At minimum, search for the original query
                fallback_search_results = self.perform_web_search(fallback_queries)
                if (
                    fallback_search_results
                    and "Web Search Results" in fallback_search_results
                ):
                    fallback_content, fallback_sources = self.extract_and_fetch_urls(
                        fallback_search_results
                    )
                    for source in fallback_sources:
                        if "url" in source:
                            self.citation_counter += 1
                            self.citations[self.citation_counter] = source
                    print(f"  - Fallback search found {len(fallback_sources)} sources")

            # PHASE 2: ITERATIVE DENOISING LOOP
            for iteration in range(self.max_iterations):
                self.research_state["iteration"] = iteration + 1
                print(
                    f"TTD-DR: Denoising iteration {iteration + 1}/{self.max_iterations}"
                )

                # STEP 1: Analyze current draft for gaps (draft-guided search)
                print("  - Analyzing draft gaps...")
                gaps = self.analyze_draft_gaps(self.current_draft, initial_query)
                self.gap_analysis_history.append(gaps)

                if not gaps:
                    print("  - No significant gaps found, research complete")
                    break

                # STEP 2: Perform gap-targeted retrieval
                print(f"  - Performing targeted search for {len(gaps)} gaps...")
                retrieval_content = self.perform_gap_targeted_search(gaps)

                # STEP 3: Extract and fetch URLs from search results
                print("  - Extracting and fetching content...")
                content_with_urls, sources = self.extract_and_fetch_urls(
                    retrieval_content
                )

                # Register sources for citations
                for source in sources:
                    if "url" in source:
                        self.citation_counter += 1
                        self.citations[self.citation_counter] = source

                # STEP 4: DENOISING - Integrate retrieved info with current draft
                print("  - Performing denoising step...")
                previous_draft = self.current_draft
                self.current_draft = self.denoise_draft_with_retrieval(
                    self.current_draft, content_with_urls, initial_query
                )
                self.draft_history.append(self.current_draft)

                # STEP 5: Evaluate quality improvement
                print("  - Evaluating draft quality...")
                quality_scores = self.evaluate_draft_quality(
                    self.current_draft, previous_draft, initial_query
                )

                # STEP 6: Component self-evolution based on feedback
                self.update_component_fitness(quality_scores)

                # STEP 7: Check termination conditions
                completeness = quality_scores.get("completeness", 0.0)
                improvement = quality_scores.get("improvement", 0.0)

                print(
                    f"  - Quality scores: Completeness={completeness:.2f}, Improvement={improvement:.2f}"
                )

                # Terminate if high quality achieved or minimal improvement
                # More lenient termination to ensure complete research
                if completeness > 0.9 or (improvement < 0.03 and completeness > 0.7):
                    print("  - Quality threshold reached, research complete")
                    break

            # PHASE 3: FINALIZATION - Polish the final draft
            print("TTD-DR: Finalizing research report...")

            # Ensure we have gathered some sources
            if len(self.citations) == 0:
                print("⚠️  Warning: No external sources found during research!")
                print("   Deep research should always consult external sources.")
            else:
                print(f"✅ Research completed with {len(self.citations)} sources")

            final_report = self.finalize_research_report(
                system_prompt, initial_query, self.current_draft
            )

            return final_report, self.total_tokens

        finally:
            # Clean up browser session
            if self.session_manager:
                print(f"🏁 Closing research session: {self.session_id}")
                close_session(self.session_id)
                self.session_manager = None

    def finalize_research_report(
        self, system_prompt: str, original_query: str, final_draft: str
    ) -> str:
        """
        Apply final polishing to the research report
        """
        # Build citation context
        citation_context = "\n\n**AVAILABLE CITATIONS:**\n"
        for num, source in self.citations.items():
            citation_context += f"[{num}] {source.get('title', 'Untitled')}\n"

        finalization_prompt = f"""
        Apply final polishing to this research report. This is the last step in the TTD-DR diffusion process.

        Original Query: {original_query}

        Current Draft:
        {final_draft}
        {citation_context}

        FINALIZATION TASKS:
        1. Ensure professional academic formatting with clear sections
        2. **CRITICAL**: Verify all citations are properly formatted as [1], [2], etc. and ADD MISSING CITATIONS
        3. Add citations to any factual claims, statistics, or findings that lack them
        4. Add a compelling title and executive summary
        5. Ensure smooth transitions between sections
        6. Add conclusion that directly addresses the original query
        7. **CRITICAL**: Remove ALL [NEEDS RESEARCH], [SOURCE NEEDED], and similar placeholder tags
        8. Replace any remaining placeholders with actual content or remove incomplete sections
        9. Polish language and style for clarity and impact

        **CRITICAL CITATION REQUIREMENTS**:
        - Every major factual claim MUST have a citation
        - Statistics, data points, and research findings MUST be cited
        - If information came from research, it needs a citation from the available sources above
        - Aim for at least 60-70% of substantive claims to have proper citations
        - Remove or rephrase claims that cannot be supported with available citations

        **CRITICAL QUALITY REQUIREMENTS**:
        - The final report must NOT contain ANY placeholder tags: [NEEDS RESEARCH], [SOURCE NEEDED], [Placeholder for...], etc.
        - Remove incomplete "Research Questions for Investigation" sections with unanswered questions
        - Do not include citation placeholders like "[1] [Placeholder for specific research citation]"
        - If sections are incomplete, either complete them with available information or remove them entirely
        - Ensure all statements are backed by available evidence or are clearly marked as preliminary findings
        - The report must be publication-ready with no incomplete elements
        - DO NOT create a References section - it will be added automatically

        Return the final polished research report with comprehensive citations.
        """

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": finalization_prompt},
                ],
                temperature=0.5,
                max_tokens=3000,
            )

            if not response.choices:
                raise ValueError(
                    "LLM returned empty response in finalize_research_report"
                )
            polished_report = response.choices[0].message.content.strip()
            polished_report = clean_reasoning_tags(polished_report)

            # Final cleanup: Remove any remaining placeholder tags
            polished_report = self.cleanup_placeholder_tags(polished_report)

            # Validate report completeness
            validation = validate_report_completeness(polished_report)

            if not validation["is_complete"]:
                print(
                    f"⚠️  Report validation found {len(validation['issues'])} issues:"
                )
                for issue in validation["issues"]:
                    print(f"   - {issue}")

                # Attempt to fix incomplete report
                polished_report = self.fix_incomplete_report(
                    polished_report, validation, original_query
                )
            else:
                print("✅ Report validation passed - report is complete")

            self.total_tokens += response.usage.completion_tokens

            # Remove any References section the LLM might have created
            polished_report = re.sub(
                r"##\s*References.*?(?=##|\Z)", "", polished_report, flags=re.DOTALL
            )
            polished_report = re.sub(
                r"(?m)^References\s*\n\s*(?:\[\d+\]\s*\n)+", "", polished_report
            )
            polished_report = re.sub(
                r"\n\s*\n\s*\n+", "\n\n", polished_report
            )  # Clean up extra newlines

            # Validate citation usage before adding references
            citation_validation = validate_citation_usage(
                polished_report, len(self.citations)
            )
            print("📊 Citation Statistics:")
            print(
                f"   - Used citations: {citation_validation['citations_used']}/{citation_validation['citations_total']}"
            )
            print(
                f"   - Usage percentage: {citation_validation['usage_percentage']:.1f}%"
            )

            if "warning" in citation_validation:
                print(f"⚠️  {citation_validation['warning']}")
                if len(citation_validation["unused_citations"]) > 0:
                    print(
                        f"   - Unused citations: {citation_validation['unused_citations'][:10]}"
                        + (
                            f"... and {len(citation_validation['unused_citations']) - 10} more"
                            if len(citation_validation["unused_citations"]) > 10
                            else ""
                        )
                    )

            # Add references section (only for citations that are actually used)
            references = "\n\n## References\n\n"
            used_citations = set(citation_validation["used_citations"])

            for num, source in sorted(self.citations.items()):
                # Only include citations that are actually used in the text
                if num in used_citations:
                    title = source.get("title", "Untitled")
                    url = source["url"]
                    access_date = source.get(
                        "access_date", datetime.now().strftime("%Y-%m-%d")
                    )
                    references += f"[{num}] {title}. Available at: <{url}> [Accessed: {access_date}]\n\n"

            # Add TTD-DR metadata
            metadata = "\n---\n\n**TTD-DR Research Metadata:**\n"
            metadata += "- Algorithm: Test-Time Diffusion Deep Researcher\n"
            metadata += f"- Denoising iterations: {len(self.draft_history) - 1}\n"
            metadata += f"- Total gaps addressed: {sum(len(gaps) for gaps in self.gap_analysis_history)}\n"
            metadata += f"- Total sources consulted: {len(self.citations)}\n"
            metadata += f"- Citations used in text: {citation_validation['citations_used']} ({citation_validation['usage_percentage']:.1f}%)\n"
            metadata += (
                f"- Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            metadata += f"- Total tokens used: {self.total_tokens}\n"

            return polished_report + references + metadata

        except Exception as e:
            return f"Finalization failed: {str(e)}\n\nReturning current draft:\n{final_draft}"
