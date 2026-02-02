from typing import Tuple, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from agents.base import BaseAgent
from config import Config
from vector_stores.chroma import ChromaDBManager
from pydantic import BaseModel, Field

class SEOMetadata(BaseModel):
    title: str = Field(description="SEO optimized title (max 60 chars)")
    meta_description: str = Field(description="Meta description (max 160 chars)")
    keywords_used: list[str] = Field(description="List of keywords actually inserted")
    confidence: float = Field(description="Confidence score 0-1")
    url_slug: str = Field(description="Recommended URL slug")

class SEOAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="SEO", temperature=Config.SEO_TEMP)
        
        self.db = ChromaDBManager()
        self.parser = JsonOutputParser(pydantic_object=SEOMetadata)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert SEO Specialist. Your goal is to optimize content for search engines without sacrificing readability.
            
            Inputs:
            1. Content (needs optimization)
            2. Target Keywords (from brief/research)
            3. Competitor Data (context)
            
            Instructions:
            - Analyze the content for keyword placement.
            - Generate optimized Title Tag and Meta Description.
            - Suggest a clean URL slug.
            - Ensure H1/H2 tags use keywords naturally.
            - Output the FINAL optimized content (Markdown) AND the Metadata (JSON).
            
            Format your response strictly as a JSON object with two keys:
            - "optimized_content": The full markdown string of the content.
            - "metadata": The object matching the schema below.
            
            {format_instructions}
            """),
            ("user", """
            Brief Keywords: {keywords}
            
            Competitor Data: {competitor_data}
            
            Content to Optimize:
            {content}
            """)
        ])
        
        # Override chain to just return the json result directly for now
        self.chain = self.prompt | self.llm | self.parser

    def optimize(self, content: str, brief: Dict) -> Tuple[str, Dict]:
        """
        Optimizes content for SEO.
        Returns: (Optimized Content String, Metadata Dictionary)
        """
        # Get keywords from brief
        keywords = brief.get("seo_keywords", [])
        keywords_str = ", ".join(keywords) if isinstance(keywords, list) else str(keywords)
        
        # Retrieve competitor info from 'seo' collection
        seo_docs = self.db.query("seo", keywords_str, k=2)
        competitor_data = "\n".join([d.page_content for d in seo_docs])
        
        input_data = {
            "content": content,
            "keywords": keywords_str,
            "competitor_data": competitor_data,
            "format_instructions": self.parser.get_format_instructions()
        }
        
        result = self.invoke(input_data)
        
        # Parse result
        final_content = result.get("optimized_content", content)
        metadata = result.get("metadata", {})
        
        # Ensure we return expected types
        return final_content, metadata
