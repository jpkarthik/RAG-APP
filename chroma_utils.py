import chromadb
from sentence_transformers import SentenceTransformer
import PyPDF2
import os
from dotenv import load_dotenv
import hashlib
import numpy as np

# Load .env file
load_dotenv()
PDF_Directory = os.getenv("PDF_Directory")
# Initialize persistent ChromaDB client
chroma_client = chromadb.PersistentClient(path=os.getenv("CHROMA_DB_PATH"))
embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_pdf_hash(pdf_file):
    try:
        hasher = hashlib.md5()
        hasher.update(pdf_file.read())
        return hasher.hexdigest()
    except Exception as e:
        print(f"Error generating hash for PDF: {str(e)}")
        return None

def extract_pdf_text_with_pages(pdf_file):
    try:
        pdf_file.seek(0)
        reader = PyPDF2.PdfReader(pdf_file)
        page_texts = []
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text() or ""
            text = text.strip()
            page_texts.append((text, page_num + 1))
            if not text:
                print(f"Warning: Page {page_num + 1} has no extractable text (possibly image-based content like graphs).")
            else:
                print(f"Page {page_num + 1} text length: {len(text)} characters")
        return page_texts
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return []

def chunk_text(page_texts, max_length=500, overlap=100):
    chunks = []
    chunk_metadata = []
    current_chunk = []
    current_length = 0
    current_page_numbers = []
    word_count = 0

    for text, page_num in page_texts:
        words = text.split()
        for word in words:
            current_length += len(word) + 1
            current_chunk.append(word)
            word_count += 1
            if not current_page_numbers or current_page_numbers[-1] != page_num:
                current_page_numbers.append(page_num)

            if word_count >= max_length:
                chunk_text = " ".join(current_chunk)
                chunks.append(chunk_text)
                chunk_metadata.append({"page_numbers": current_page_numbers.copy()})
                current_chunk = current_chunk[-overlap:] if overlap < len(current_chunk) else current_chunk
                current_length = sum(len(w) + 1 for w in current_chunk)
                word_count = len(current_chunk)
                current_page_numbers = current_page_numbers[-1:] if current_page_numbers else []
        if not current_chunk:
            current_page_numbers = []

    if current_chunk:
        chunk_text = " ".join(current_chunk)
        if chunk_text.strip():
            chunks.append(chunk_text)
            chunk_metadata.append({"page_numbers": current_page_numbers})

    return chunks, chunk_metadata

def add_documents(pdf_file, filename):
    try:
        filename = os.path.normpath(filename)
        pdf_file.seek(0)
        pdf_hash = get_pdf_hash(pdf_file)
        if not pdf_hash:
            print(f"Failed to generate hash for {filename}")
            return None, False
        
        collection_name = f"pdf_{pdf_hash}"
        collection = chroma_client.get_or_create_collection(collection_name)
        
        if collection.count() > 0:
            print(f"Collection {collection_name} already exists with {collection.count()} documents")
            return collection, False
        
        page_texts = extract_pdf_text_with_pages(pdf_file)
        if not page_texts:
            print(f"No text extracted from {filename}. If it contains images (e.g., graphs), consider OCR (e.g., pytesseract).")
            return collection, False
        
        chunks, chunk_metadata = chunk_text(page_texts, max_length=500, overlap=100)
        if not chunks:
            print(f"No chunks created for {filename}")
            return collection, False
        
        print(f"Created {len(chunks)} chunks for {filename}: {[len(c) for c in chunks]}")
        ids = [f"{pdf_hash}_{i}" for i in range(len(chunks))]
        embeddings = embedder.encode(chunks, show_progress_bar=True).tolist()
        embedding_norms = [np.linalg.norm(emb) for emb in embeddings]
        print(f"Embedding norms for {filename}: {embedding_norms}")
        metadatas = [
            {"chunk_id": id, "page_numbers": ",".join(str(p) for p in meta["page_numbers"]), "filename": filename}
            for id, meta in zip(ids, chunk_metadata)
        ]
        
        try:
            collection.add(
                documents=chunks,
                embeddings=embeddings,
                ids=ids,
                metadatas=metadatas
            )
            print(f"Successfully added {len(chunks)} chunks to collection {collection_name}")
            return collection, True
        except Exception as e:
            print(f"Error adding to ChromaDB for {filename}: {str(e)}")
            return collection, False
    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return None, False

def load_pdfs_from_directory(directory):
    documents = []
    ids = []
    collections = []
    default_pdf = PDF_Directory
    try:
        directory = os.path.normpath(directory)
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist, attempting to load default PDF.")
        else:
            for filename in os.listdir(directory):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(directory, filename)
                    try:
                        with open(file_path, "rb") as file:
                            collection, processed = add_documents(file, filename)
                            if collection is None:
                                continue
                            if processed:
                                file.seek(0)
                                chunks, chunk_metadata = chunk_text(extract_pdf_text_with_pages(file))
                                ids_chunk = [f"{get_pdf_hash(file)}_{j}" for j in range(len(chunks))]
                                documents.extend(chunks)
                                ids.extend(ids_chunk)
                            collections.append(collection)
                    except (FileNotFoundError, OSError) as e:
                        print(f"File {file_path} not found or invalid: {str(e)}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        if not collections and os.path.exists(default_pdf):
            print(f"No collections found, loading default PDF: {default_pdf}")
            for filename in os.listdir(default_pdf):
                if filename.lower().endswith(".pdf"):
                    file_path = os.path.join(default_pdf, filename)
                    try:
                        with open(file_path, "rb") as file:
                            collection, processed = add_documents(file, filename)
                            if collection is None:
                                continue
                            if processed:
                                file.seek(0)
                                chunks, chunk_metadata = chunk_text(extract_pdf_text_with_pages(file))
                                ids_chunk = [f"{get_pdf_hash(file)}_{j}" for j in range(len(chunks))]
                                documents.extend(chunks)
                                ids.extend(ids_chunk)
                            collections.append(collection)
                    except (FileNotFoundError, OSError) as e:
                        print(f"File {file_path} not found or invalid: {str(e)}")
                    except Exception as e:
                        print(f"Error processing {file_path}: {str(e)}")
        
        elif not collections:
            print("No valid PDFs found or processed, and default PDF is unavailable.")
        else:
            print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        return documents, ids, collections
    except Exception as e:
        print(f"Error accessing directory {directory}: {str(e)}")
        if os.path.exists(default_pdf):
            print(f"Falling back to default PDF: {default_pdf}")
            with open(default_pdf, "rb") as file:
                collection, processed = add_documents(file, os.path.basename(default_pdf))
                if collection and processed:
                    collections.append(collection)
                    file.seek(0)
                    chunks, chunk_metadata = chunk_text(extract_pdf_text_with_pages(file))
                    ids_chunk = [f"{get_pdf_hash(file)}_{j}" for j in range(len(chunks))]
                    documents.extend(chunks)
                    ids.extend(ids_chunk)
        return [], [], collections

def query_collections(query, collections, top_k=3):
    try:
        print(f"Query: {query}")
        query_embedding = embedder.encode([query]).tolist()
        query_norm = np.linalg.norm(query_embedding[0])
        print(f"Query embedding norm: {query_norm}")
        results = []
        for collection in collections:
            query_results = collection.query(
                query_embeddings=query_embedding,
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            documents = query_results["documents"][0]
            metadatas = query_results["metadatas"][0]
            distances = query_results["distances"][0]
            print(f"Raw distances for collection {collection.name}: {distances}")
            print(f"Document lengths: {[len(doc) for doc in documents]}")
            print(f"Document previews: {[doc[:50] + '...' for doc in documents]}")
            similarities = [1 - dist / 2 if dist is not None else 0 for dist in distances]
            print(f"Calculated similarities (1 - dist/2): {[1 - d / 2 for d in distances]}")
            print(f"Normalized similarities: {similarities}")
            parsed_results = [
                {
                    "document": doc,
                    "metadata": {
                        "chunk_id": meta["chunk_id"],
                        "page_numbers": [int(p) for p in meta["page_numbers"].split(",") if p],
                        "filename": meta.get("filename", "unknown")
                    },
                    "similarity": sim
                }
                for doc, meta, sim in zip(documents, metadatas, similarities)
            ]
            results.extend(parsed_results)
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)[:top_k]
        if not results:
            print("No results returned from query.")
        return results
    except Exception as e:
        print(f"Error querying collections: {str(e)}")
        return []