from typing import Dict, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.base import BaseAgent
from config import Config
from vector_stores.chroma import ChromaDBManager

class WriterAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Writer", temperature=Config.WRITER_TEMP)
        
        # We can use RAG here to find writing samples if needed
        self.db = ChromaDBManager()
        
        self.parser = StrOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Content Writer. Your goal is to write high-quality, engaging content based on a strict brief and research.
            
            Inputs:
            1. Content Brief: Contains the topic, audience, tone, and outline.
            2. Research Summary: Contains the factual information to include.
            
            Instructions:
            - Follow the outline in the brief EXACTLY.
            - Adopt the specified tone and voice.
            - Integrate the research findings naturally.
            - Use markdown formatting (H1 for title, H2 for main sections).
            - Do not invent facts; rely on the research provided.
            
            Write the full article now.
            """),
            ("user", """
            Brief:
            {brief}
            
            Research Findings:
            {research}
            """)
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def write(self, brief: Dict, research: str) -> str:
        """
        Generates content draft based on brief and research.
        """
        # Convert brief dict to string representation for the prompt
        brief_str = str(brief)
        
        # Optional: Retrieve a writing sample to guide style (simplistic implementation for now)
        # style_docs = self.db.query("writing", "style guide", k=1)
        # style_context = style_docs[0].page_content if style_docs else ""
        
        input_data = {
            "brief": brief_str,
            "research": research
        }
        
        return self.invoke(input_data)
