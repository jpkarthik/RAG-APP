from chroma_utils import query_collections
from openai import OpenAI
from dotenv import load_dotenv
import os
from collections import deque
import json
import re

# Load .env file
load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

class AgenticRAG:
    def __init__(self, max_history=5):
        """
        Initializes Agentic RAG with optional history tracking.
        
        Args:
            max_history (int): Maximum number of conversation turns to store (default: 5).
        """
        self.history = deque(maxlen=max_history)  # Stores (query, response) pairs; remove if not needed

    def add_to_history(self, query, response):
        """
        Adds query and response to the conversation history.
        
        Args:
            query (str): User query.
            response (str): RAG response (raw or fine-tuned).
        """
        self.history.append((query, response))

    def get_history_context(self):
        """
        Constructs a string of conversation history for context.
        
        Returns:
            str: Formatted history string.
        """
        if not self.history:
            return ""
        context = "Conversation History:\n"
        for i, (query, response) in enumerate(self.history, 1):
            context += f"Turn {i} - Query: {query}\nResponse: {response}\n\n"
        return context

    def analyze_query(self, query):
        """
        Analyzes query complexity and suggests retrieval strategy.
        
        Args:
            query (str): User question.
        
        Returns:
            tuple: (is_complex, sub_queries) where is_complex is bool and sub_queries is list of str.
        """
        print(f"Analyzing Query: {query}")
        # Simple heuristic: Split by 'and' or multi-sentence queries
        if "and" in query.lower() or query.count(".") > 1:
            print("Query identified as complex; splitting into sub-queries.")
            sub_queries = re.split(r'[.;]\s*and\s*|[.;]\s*', query)
            sub_queries = [q.strip() for q in sub_queries if q.strip()]
            return True, sub_queries
        print("Query identified as simple; proceeding with direct retrieval.")
        return False, [query]

    def agentic_rag(self, query, collections, top_k=3, fine_tune=False):
        """
        Performs Agentic RAG with dynamic query handling and structured output.
        
        Args:
            query (str): User question.
            collections (list): List of ChromaDB collections (one per PDF).
            top_k (int): Number of top relevant chunks to retrieve (default: 3).
            fine_tune (bool): If True, use LLM for structured output; if False, direct JSON structuring (default: False).
        
        Returns:
            str: Agent reasoning and structured JSON response.
        """
        try:
            if not collections:
                return "No document collections available to query."
            if not query:
                return "No query provided."
            
            # Agent reasoning
            is_complex, sub_queries = self.analyze_query(query)
            all_results = []
            reasoning_steps = ["Agent Reasoning: Initial query analysis completed."]
            
            if is_complex:
                reasoning_steps.append(f"Query split into {len(sub_queries)} sub-queries: {sub_queries}")
                for sub_query in sub_queries:
                    results = query_collections(sub_query, collections, top_k)
                    if results:
                        all_results.extend(results)
                        reasoning_steps.append(f"Retrieved {len(results)} chunks for sub-query: {sub_query}")
                    else:
                        reasoning_steps.append(f"No results for sub-query: {sub_query}")
            else:
                results = query_collections(query, collections, top_k)
                if results:
                    all_results.extend(results)
                    reasoning_steps.append(f"Retrieved {len(results)} chunks for query: {query}")
                else:
                    reasoning_steps.append(f"No results for query: {query}")
            
            if not all_results:
                return "\n".join(reasoning_steps) + "\nNo relevant documents found."

            # Deduplicate and sort by similarity
            chunk_ids = set()
            unique_results = []
            for result in all_results:
                if result["metadata"]["chunk_id"] not in chunk_ids:
                    chunk_ids.add(result["metadata"]["chunk_id"])
                    unique_results.append(result)
            unique_results = sorted(unique_results, key=lambda x: x["similarity"], reverse=True)[:top_k]

            # Construct context and raw response
            raw_response = ""
            context = ""
            print("\nRetrieved Chunks:")
            for i, result in enumerate(unique_results):
                doc = result["document"]
                metadata = result["metadata"]
                similarity = result["similarity"]
                page_numbers = metadata.get("page_numbers", [])
                filename = metadata.get("filename", "unknown")
                page_ref = f"Pages {', '.join(map(str, page_numbers))}" if page_numbers else "No page info"
                # print(f"Chunk {i+1} (Cosine Similarity: {similarity:.3f}, {page_ref}, PDF: {filename}):")
                # print(f"{doc[:200]}..." if len(doc) > 200 else doc)
                # print("-" * 50)
                chunk_text = f"Chunk {i+1} (Similarity: {similarity:.3f}, {page_ref}, PDF: {filename}):\n{doc}\n"
                raw_response += chunk_text
                context += f"Document {i+1} ({page_ref}, PDF: {filename}):\n{doc}\n\n"
            
            # Direct structuring when fine_tune=False
            if not fine_tune:
                best_result = max(unique_results, key=lambda x: x["similarity"])
                summary = best_result["document"][:200] + "..." if len(best_result["document"]) > 200 else best_result["document"]
                all_pages = [page for result in unique_results for page in result["metadata"]["page_numbers"]]
                source = best_result["metadata"]["filename"]
                structured_response = {
                    "question": query,
                    "answer": {
                        "summary": summary,
                        "details": raw_response,
                        "pages": list(set(all_pages)),
                        "source": source
                    }
                }
                self.add_to_history(query, json.dumps(structured_response))  # Remove if disabling history
                return "\n".join(reasoning_steps) + f"\n\nStructured Output RAG Response:\n{json.dumps(structured_response, indent=2)}"

            # LLM structuring when fine_tune=True
            history_context = self.get_history_context()  # Remove if disabling history
            prompt = f"""Conversation History:
                            {history_context}

                            Retrieved Context:
                            {context}

                            Current Question: {query}
                            Provide a structured JSON response with the following format:
                            {{
                            "question": "<the original question>",
                            "answer": {{
                                "summary": "<a concise answer based on the context>",
                                "details": "<detailed explanation or list of points>",
                                "pages": "<list of relevant page numbers>",
                                "source": "<filename of the PDF>"
                            }}
                            }}
                            Include reasoning steps if the query was complex.
                              Ensure the response is valid JSON and references the retrieved context accurately.
                                Note that some PDFs may contain images (e.g., graphs) not included in the context:
                        """
            
          

            # Generate structured response using OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            structured_answer = response.choices[0].message.content.strip()
            print(f"Raw LLM Response: {structured_answer}")  # Debug log
            
            # Extract JSON using regex to handle potential extra text
            json_match = re.search(r'\{.*\}', structured_answer, re.DOTALL)
            if json_match:
                structured_answer = json_match.group(0).strip()
            
            try:
                # Validate JSON
                parsed_json = json.loads(structured_answer)
                self.add_to_history(query, structured_answer)  # Remove if disabling history
                return "\n".join(reasoning_steps) + f"\n\nStructured Output RAG Response:\n{json.dumps(parsed_json, indent=2)}"
            except json.JSONDecodeError as e:
                return "\n".join(reasoning_steps) + f"\n\nError: Invalid JSON response from LLM:\n{structured_answer}\nError Details: {str(e)}"
        
        except Exception as e:
            return f"Error processing query: {str(e)}"