from graph.state import ContentState

def should_retry_writing(state: ContentState) -> str:
    """Check if draft needs rewriting"""
    draft = state.get("draft_content", "")
    brief = state.get("brief", {})
    
    # Simple check for empty draft
    if not draft:
        return "rewrite"
        
    # In a real implementation, we would check word counts or other metrics
    target_words = brief.get("word_count", 1500)
    actual_words = len(draft.split())
    
    # Demo logic: 
    # If word count is drastically off (e.g. < 10% of target), retry
    # For now, we'll assume the mock always writes enough.
    
    return "proceed"

def should_retry_editing(state: ContentState) -> str:
    """Check editing quality"""
    confidence = state.get("confidence_scores", {}).get("editing", 1.0)
    
    if confidence < 0.7:
        return "re_edit"
    
    return "proceed"

def check_errors(state: ContentState) -> str:
    """Check for fatal errors"""
    if state.get("errors"):
        return "error"
    
    return "continue"
