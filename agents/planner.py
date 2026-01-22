from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.base import BaseAgent
from models import ContentBrief
from config import Config

class PlannerAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Planner", temperature=Config.PLANNER_TEMP)
        self.parser = JsonOutputParser(pydantic_object=ContentBrief)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Content Strategist. Your goal is to plan high-quality, SEO-optimized content.
            
            Given a content request, you must create a detailed brief.
            
            Analyze the request for:
            1. User Intent: What do they actually want?
            2. Target Audience: Who is this for?
            3. Tone/Voice: What is the appropriate style?
            
            Output strictly valid JSON that matches this schema:
            {format_instructions}
            """),
            ("user", "{content_request}")
        ])
        
        # Build chain: Prompt -> LLM -> JSON Parser
        self.chain = self.prompt | self.llm | self.parser

    def plan(self, content_request: str) -> dict:
        """
        Generates a content brief from a user request.
        """
        input_data = {
            "content_request": content_request,
            "format_instructions": self.parser.get_format_instructions()
        }
        return self.invoke(input_data)
