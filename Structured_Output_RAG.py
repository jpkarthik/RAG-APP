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

class StructuredOutputRAG:
    def __init__(self, max_history=5):
        """
        Initializes Structured Output RAG with optional history tracking.
        
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

    def structured_output_rag(self, query, collections, top_k=3, fine_tune=False):
        """
        Performs Structured Output RAG with toggleable direct or LLM-based JSON structuring.
        
        Args:
            query (str): User question.
            collections (list): List of ChromaDB collections (one per PDF).
            top_k (int): Number of top relevant chunks to retrieve (default: 3).
            fine_tune (bool): If True, use LLM for structured output; if False, direct JSON structuring (default: False).
        
        Returns:
            str: Structured JSON response (direct or LLM-generated).
        """
        try:
            if not collections:
                return "No document collections available to query."
            if not query:
                return "No query provided."
            
            # Retrieval
            results = query_collections(query, collections, top_k)
            if not results:
                return "No relevant documents found for the query."
            
            # Construct context and raw response
            raw_response = ""
            context = ""
            chunk_ids = set()
            print(f"\nRetrieved Chunks for Query: {query}")
            for i, result in enumerate(results):
                chunk_id = result["metadata"]["chunk_id"]
                if chunk_id in chunk_ids:
                    continue
                chunk_ids.add(chunk_id)
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
            
            if not chunk_ids:
                # self.add_to_history(query, raw_response)  # Remove if disabling history
                return f"\nRaw Structured Output RAG Response:\n{raw_response}\nNote: If relevant content is in images (e.g., graphs), consider OCR for text extraction."
            
            # Direct structuring when fine_tune=False
            if not fine_tune:
                # Use the highest similarity chunk for summary, aggregate pages and source
                best_result = max(results, key=lambda x: x["similarity"])
                summary = best_result["document"][:200] + "..." if len(best_result["document"]) > 200 else best_result["document"]
                all_pages = [page for result in results for page in result["metadata"]["page_numbers"]]
                source = best_result["metadata"]["filename"]
                structured_response = {
                    "question": query,
                    "answer": {
                        "summary": summary,
                        "details": raw_response,  # Raw chunks as details for simplicity
                        "pages": list(set(all_pages)),  # Deduplicated page numbers
                        "source": source
                    }
                }
                # self.add_to_history(query, json.dumps(structured_response))  # Remove if disabling history
                return f"\nStructured Output RAG Response:\n{json.dumps(structured_response, indent=2)}"
            
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
                            Ensure the response is valid JSON and references the retrieved context accurately.
                              Note that some PDFs may contain images (e.g., graphs) not included in the context:
                        """
        
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
                # self.add_to_history(query, structured_answer)  # Remove if disabling history
                return f"\nStructured Output RAG Response:\n{json.dumps(parsed_json, indent=2)}"
            except json.JSONDecodeError as e:
                return f"\nError: Invalid JSON response from LLM:\n{structured_answer}\nError Details: {str(e)}"
        
        except Exception as e:
            return f"Error processing query: {str(e)}"
