from typing import List, Tuple, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from agents.base import BaseAgent
from vector_stores.chroma import ChromaDBManager
from config import Config

class ResearchAgent(BaseAgent):
    def __init__(self):
        super().__init__(name="Research", temperature=Config.RESEARCHER_TEMP)
        
        # Initialize Vector DB access
        self.db = ChromaDBManager()
        
        # Output parser
        self.parser = StrOutputParser()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Research Analyst. Your goal is to provide accurate, factual information based ONLY on the provided context.
            
            You will be given a set of queries and the retrieved documents associated with them.
            
            Synthesize these findings into a coherent research summary.
            - Cite your sources where possible (using 'Source: [Title]').
            - If the information is conflicting, note the discrepancy.
            - If the retrieved documents do not answer the query, state: "Insufficient information in current knowledge base."
            
            Format your output as markdown.
            """),
            ("user", """
            Queries: {queries}
            
            Retrieved Context:
            {context}
            """)
        ])
        
        self.chain = self.prompt | self.llm | self.parser

    def research(self, queries: List[str]) -> Tuple[str, List[Dict]]:
        """
        Conducts research by querying the vector store and synthesizing findings.
        
        Returns:
            Tuple containing:
            1. Synthesized summary string
            2. List of unique retrieved documents (dictionaries)
        """
        all_docs = []
        context_parts = []
        
        # 1. Retrieve documents for each query
        for q in queries:
            try:
                # Query the 'research' collection
                results = self.db.query_multireturn("research", q, k=Config.RETRIEVAL_K)
                
                if results:
                    context_parts.append(f"--- Results for query: '{q}' ---")
                    for r in results:
                        # Format context for the LLM
                        source = r.get("metadata", {}).get("title") or r.get("source", "Unknown")
                        content = r.get("content", "").strip()
                        context_parts.append(f"Source: {source}\nContent: {content}\n")
                        
                        # Add to unique docs list (deduplicating by content content/source somewhat crudely)
                        if r not in all_docs:
                            all_docs.append(r)
            except Exception as e:
                print(f"[{self.name}] Error querying DB for '{q}': {e}")
        
        if not context_parts:
            return "No relevant documents found in the knowledge base.", []
            
        full_context = "\n".join(context_parts)
        
        # 2. Synthesize with LLM
        input_data = {
            "queries": "\n- ".join(queries),
            "context": full_context
        }
        
        summary = self.invoke(input_data)
        
        return summary, all_docs
