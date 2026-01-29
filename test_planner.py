import os
from dotenv import load_dotenv
from agents.planner import PlannerAgent
import json

# Ensure env vars are loaded
load_dotenv()

def test_planner():
    print("Initializing Planner Agent...")
    try:
        planner = PlannerAgent()
    except Exception as e:
        print(f"Failed to initialize agent: {e}")
        return

    topic = "A guide to growing tomatoes in small apartments"
    print(f"\nRunning plan for topic: '{topic}'...\n")
    
    try:
        brief = planner.plan(topic)
        print("Successfully generated brief:")
        print(json.dumps(brief, indent=2))
    except Exception as e:
        print(f"Error during planning: {e}")

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY not found in environment variables.")
        print("Please create a .env file with your API key.")
    else:
        test_planner()
