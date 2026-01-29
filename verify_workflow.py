from graph.workflow import create_content_workflow
import json

def main():
    print("Initializing workflow...")
    app = create_content_workflow()
    
    request = "Write a short blog post about the benefits of green tea."
    initial_state = {
        "content_request": request,
        "settings": {
            "word_count": 500,
            "tone": "informative"
        }
    }
    
    print(f"Invoking workflow with request: '{request}'")
    try:
        result = app.invoke(initial_state)
        
        print("\n--- Workflow Completed Successfully ---")
        print(f"Final Content Preview: {result.get('final_content', '')[:100]}...")
        
        print("\nAgent Logs:")
        for log in result.get('agent_logs', []):
            print(f"- {log.get('agent')}: {json.dumps(log, default=str)}")

        if result.get('errors'):
            print("\nErrors encountered:")
            for error in result['errors']:
                print(f"- {error}")
            
    except Exception as e:
        print(f"\nWorkflow FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
