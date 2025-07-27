from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

docs = [
    "Transformers are devices that transfer electrical energy between circuits.",
    "They work based on electromagnetic induction.",
    "Primary and secondary coils have different turn ratios.",
    "Voltage and current vary with the coil ratios.",
    "Core materials affect efficiency.",
    "Oil-cooled transformers are common in power grids.",
    "Isolation transformers separate circuits safely.",
    "Step-up transformers increase voltage.",
    "Step-down transformers decrease voltage.",
    "Efficiency depends on load and design."
]

chunks = [doc.split(". ") for doc in docs]
flattened_chunks = [chunk for sublist in chunks for chunk in sublist if chunk]


model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(flattened_chunks, convert_to_tensor=True)



def query_rag_topk(question, chunks, embeddings, k=3):
    question_embedding = model.encode([question], convert_to_tensor=True)
    scores = cosine_similarity(question_embedding.cpu().numpy(), embeddings.cpu().numpy())[0]
    top_k_indices = np.argsort(scores)[-k:][::-1]
    top_chunks = [chunks[i] for i in top_k_indices]
    return f"Top relevant info:\n" + "\n".join(f"- {chunk}" for chunk in top_chunks)

# Example query
query = "How does voltage change in transformers?"
print(f"User Query: {query}")
response = query_rag_topk(query, flattened_chunks, embeddings)
print(response)