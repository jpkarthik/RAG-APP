from chroma_utils import query_collections
from openai import OpenAI
from dotenv import load_dotenv
import os
from collections import deque

# Load .env file
load_dotenv()

# Initialize OpenAI client
try:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception as e:
    raise ValueError(f"Failed to initialize OpenAI client: {str(e)}")

class MultiQueryRAG:
    def __init__(self, max_history=5):
        """
        Initializes Multi-Query RAG with history tracking.
        
        Args:
            max_history (int): Maximum number of conversation turns to store (default: 5).
        """
        self.history = deque(maxlen=max_history)  # Stores (queries, response) pairs

    def add_to_history(self, queries, response):
        """
        Adds queries and response to the conversation history.
        
        Args:
            queries (list): List of user queries.
            response (str): RAG response (raw or fine-tuned).
        """
        self.history.append((queries, response))

    def get_history_context(self):
        """
        Constructs a string of conversation history for context.
        
        Returns:
            str: Formatted history string.
        """
        if not self.history:
            return ""
        context = "Conversation History:\n"
        for i, (queries, response) in enumerate(self.history, 1):
            context += f"Turn {i} - Queries: {', '.join(queries)}\nResponse: {response}\n\n"
        return context

    def multi_query_rag(self, queries, collections, top_k=3, fine_tune=False):
        """
        Performs Multi-Query RAG, querying ChromaDB collections for multiple queries.
        
        Args:
            queries (list): List of user questions.
            collections (list): List of ChromaDB collections to query.
            top_k (int): Number of top relevant chunks per query (default: 3).
            fine_tune (bool): If True, fine-tune with OpenAI (default: False).
        
        Returns:
            str: Raw or fine-tuned RAG response with page references.
        """
        try:
            if not collections:
                return "No document collections available to query."
            if not queries:
                return "No queries provided."
            
            # Query collections and deduplicate chunks
            chunk_ids = set()
            all_results = []
            context = ""
            raw_response = ""
            for query in queries:
                results = query_collections(query, collections, top_k)
                if not results:
                    raw_response += f"\nQuery: {query}\nNo relevant documents found.\n"
                    continue
                
                raw_response += f"\nQuery: {query}\n"
                print(f"\nRetrieved Chunks for Query: {query}")
                query_results = []
                for i, res in enumerate(results):
                    chunk_id = res["metadata"]["chunk_id"]
                    if chunk_id in chunk_ids:
                        continue
                    chunk_ids.add(chunk_id)
                    query_results.append(res)
                    doc = res["document"]
                    similarity = res["similarity"]
                    page_numbers = res["metadata"].get("page_numbers", [])
                    page_ref = f"Pages {', '.join(map(str, page_numbers))}" if page_numbers else "No page info"
                    print(f"Chunk {i+1} (Cosine Similarity: {similarity:.3f}, {page_ref}):")
                    print(f"{doc[:200]}..." if len(doc) > 200 else doc)
                    print("-" * 50)
                    chunk_text = f"Chunk {chunk_id} (Similarity: {similarity:.3f}, {page_ref}):\n{doc}\n"
                    raw_response += chunk_text
                    context += f"Query: {query}\nDocument {chunk_id} ({page_ref}): {doc}\n\n"
                all_results.extend(query_results)
            
            if not all_results:
                self.add_to_history(queries, raw_response)
                return f"\nRaw Multi-Query RAG Response:\n\n{raw_response}"
            
            # If fine_tune is False, return raw response
            if not fine_tune:
                self.add_to_history(queries, raw_response)
                return f"\nRaw Multi-Query RAG Response:\n{raw_response}\n"
                
            # If fine_tune is True, construct prompt
            history_context = self.get_history_context()
            prompt = f"""Conversation History:
                        {history_context}

                        Retrieved Context:
                        {context}

                        Current Questions: {', '.join(queries)}
                        Provide a detailed, cohesive answer addressing all questions based on the retrieved context 
                        and conversation history, referencing relevant page numbers if applicable:
                        """
            
            # Generate response using LLM
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                
                max_tokens=1000,
                temperature=1000.7
            )
            
            answer = response.choices[0].message.content.strip()
            self.add_to_history(queries, answer)
            return f"\nFine-Tuned Multi-Query RAG Answer:\n{answer}\n"
            
        except Exception as e:
            return f"Error processing queries: {str(e)}"