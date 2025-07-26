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

class ConversationalRAG:
    def __init__(self, max_history=5):
        """
        Initializes Conversational RAG with history tracking.
        
        Args:
            max_history (int): Maximum number of conversation turns to store (default: 5).
        """
        self.history = deque(maxlen=max_history)  # Stores (query, response) pairs

    def add_to_history(self, query, response):
        """
        Adds a query and response to the conversation history.
        
        Args:
            query (str): User's query.
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

    def conversational_rag(self, query, collections, top_k=3, fine_tune=False):
        """
        Performs Conversational RAG with history, querying ChromaDB collections, and optional LLM fine-tuning.
        
        Args:
            query (str): User's question.
            collections (list): List of ChromaDB collections to query.
            top_k (int): Number of top relevant chunks to retrieve (default: 3).
            fine_tune (bool): If True, fine-tune with OpenAI (default: False).
        
        Returns:
            str: Raw or fine-tuned RAG response with page references.
        """
        try:
            if not collections:
                return "No document collections available to query."
            
            # Query collections to get relevant chunks, metadata, and similarity scores
            results = query_collections(query, collections, top_k)
            if not results:
                return "No relevant documents found for the query."
            
            # Construct raw RAG response and print retrieved chunks
            raw_response = ""
            context = ""
            print("\nRetrieved Chunks:")
            for i, result in enumerate(results):
                doc = result["document"]
                metadata = result["metadata"]
                similarity = result["similarity"]
                page_numbers = metadata.get("page_numbers", [])
                page_ref = f"Pages {', '.join(map(str, page_numbers))}" if page_numbers else "No page info"
                # print(f"Chunk {i+1} (Cosine Similarity: {similarity:.3f}, {page_ref}):")
                # print(f"{doc[:200]}..." if len(doc) > 200 else doc)
                # print("-" * 50)
                chunk_text = f"Chunk {i+1} (Similarity: {similarity:.3f}, {page_ref}):\n{doc}\n"
                raw_response += chunk_text
                context += f"Document {i+1} ({page_ref}):\n{doc}\n\n"
            
            # If fine_tune is False, add to history and return raw response
            if not fine_tune:
                self.add_to_history(query, raw_response)
                return f"\nRaw RAG Response:\n{raw_response}"
            
            # If fine_tune is True, construct prompt with history and context
            history_context = self.get_history_context()
            prompt = f"""Conversation History:
                        {history_context}

                        Retrieved Context:
                        {context}

                        Current Question: {query}
                        Provide a detailed answer based on the retrieved context and conversation history, 
                        referencing relevant page numbers if applicable:
                        """
            
            # Generate response using OpenAI
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            
            answer = response.choices[0].message.content.strip()
            self.add_to_history(query, answer)
            return f"\nFine-Tuned RAG Answer:\n{answer}"
        
        except Exception as e:
            return f"Error processing query: {str(e)}"