from langgraph.graph import StateGraph, END
from graph.state import ContentState
from graph.nodes import (
    planning_node,
    research_node,
    writing_node,
    editing_node,
    seo_node
)
from graph.edges import should_retry_writing, check_errors

def create_content_workflow():
    """Creates the LangGraph workflow"""
    
    # Initialize graph
    workflow = StateGraph(ContentState)
    
    # Add nodes
    workflow.add_node("planner", planning_node)
    workflow.add_node("researcher", research_node)
    workflow.add_node("writer", writing_node)
    workflow.add_node("editor", editing_node)
    workflow.add_node("seo", seo_node)
    
    # Set entry point
    workflow.set_entry_point("planner")
    
    # Add edges (linear flow with conditionals)
    workflow.add_edge("planner", "researcher")
    workflow.add_edge("researcher", "writer")
    
    # Conditional edge: check if writing needs retry
    workflow.add_conditional_edges(
        "writer",
        should_retry_writing,
        {
            "rewrite": "writer",  # Loop back
            "proceed": "editor"
        }
    )
    
    # TODO: Add error handling edges using check_errors if needed
    
    workflow.add_edge("editor", "seo")
    workflow.add_edge("seo", END)
    
    # Compile the graph
    app = workflow.compile()
    
    return app
