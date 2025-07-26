from chroma_utils import add_documents, load_pdfs_from_directory, query_collections
from Simple_RAG import simple_rag
from Conversational_RAG import ConversationalRAG
from Multi_Query_RAG import MultiQueryRAG
from Multi_document_RAG import MultiDocumentRAG
from Hierarchical_RAG import HierarchicalRAG
from Structured_Output_RAG import StructuredOutputRAG
from Agentic_RAG import AgenticRAG
import os

def main():
    print("Main Program")
    input_pdf_source = r"D:\PythonProject\RAG_PROJECT\pdfs\the_hindu_marriage_act_1955.pdf"
    multi_doc_directory = r"D:\PythonProject\RAG_PROJECT\pdfs"
    
    query = "Breif Proceedings to be in camera and may not be printed or published."
    k_val=3
    queries = [
                    #"Effects of Battle of Plassey",                    
                    # "on what basis divorce will be granted",
                    "What are the divorce grounds and their legal implications?"
                ]

    #TestSinglePDF(input_pdf_source)
    
    #Test_Simple_RAG(input_pdf_source,query,k,True,0,500)
    
    #Test_Conversational_RAG(input_pdf_source)

    #Test_Multi_Query_RAG(input_pdf_source,queries,fine_tune=False)

    #Test_Multi_Document_RAG(multi_doc_directory,queries,fine_tune=True,k_val=k_val)

    #Test_Hierarchical_RAG(multi_doc_directory,queries,True)

    #Test_Structured_Output_RAG(multi_doc_directory,queries,False)

    Test_Agentic_RAG(multi_doc_directory,queries,False)



def Test_Agentic_RAG(directory, queries, fine_tune=False):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = AgenticRAG(max_history=5)
           
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.agentic_rag(query, collections, top_k=3, fine_tune=fine_tune)
                print(answer)
                print("-"*200)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")

def Test_Structured_Output_RAG(directory,queries, fine_tune=False):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = StructuredOutputRAG(max_history=5)
            
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.structured_output_rag(query, collections, top_k=3, fine_tune=fine_tune)
                print(answer)
                print("-"*250)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")

def Test_Hierarchical_RAG(directory,queries, fine_tune=False):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = HierarchicalRAG(max_history=5)
           
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.hierarchical_rag(query, collections, top_k_coarse=2, top_k_fine=3, fine_tune=fine_tune)
                print(answer)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")


def Test_Multi_Document_RAG(directory,queries, fine_tune=False,k_val=3):
    try:
        directory = os.path.normpath(directory)
        documents, ids, collections = load_pdfs_from_directory(directory)
        print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
        if collections:
            rag = MultiDocumentRAG(max_history=5)
           
            for query in queries:
                print(f"\nQuery: {query}")
                answer = rag.multi_document_rag(query, collections, top_k=k_val, fine_tune=fine_tune)
                print(answer)
        else:
            print(f"No valid PDFs processed in {directory}")
    except Exception as e:
        print(f"Error processing directory {directory}: {str(e)}")


def Test_Multi_Query_RAG(pdf_path,queries, fine_tune=False):
    try:
        pdf_path = os.path.normpath(pdf_path)
        with open(pdf_path, "rb") as file:
            collection, processed = add_documents(file, os.path.basename(pdf_path))
            if collection:
                print(f"Processed: {processed}, Documents in collection: {collection.count()}")
                rag = MultiQueryRAG(max_history=5)
              
                print(f"\nQueries: {', '.join(queries)}")
                answer = rag.multi_query_rag(queries, [collection], top_k=3, fine_tune=fine_tune)
                print(answer)
            else:
                print(f"Failed to process {pdf_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"PDF file {pdf_path} not found or invalid: {str(e)}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")

def TestSinglePDF(pdf_path):
    try:
        # Normalize path for Windows
        pdf_path = os.path.normpath(pdf_path)
        with open(pdf_path, "rb") as file:
            collection, processed = add_documents(file, os.path.basename(pdf_path))
            if collection:
                print(f"Processed: {processed}, Documents in collection: {collection.count()}")
                results = query_collections("What is the Hindu Marriage Act?", [collection], top_k=3)
                for result in results:
                    print(f"Document: {result['document'][:50]}...")
                    print(f"Metadata: {result['metadata']}")
            else:
                print(f"Failed to process {pdf_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"PDF file {pdf_path} not found or invalid: {str(e)}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")


def Test_Simple_RAG(pdf_path,query,k,finetune,temperature,maxtokens):
    try:
        # Normalize path for Windows
        pdf_path = os.path.normpath(pdf_path)
        with open(pdf_path, "rb") as file:
            collection, processed = add_documents(file, os.path.basename(pdf_path))
            if collection:
                print(f"Processed: {processed}, Documents in collection: {collection.count()}")
                # Test Simple RAG
                answer = simple_rag(query, [collection], top_k=k,fine_tune=finetune,tempr_val=temperature,maxtoken=maxtokens)
                print(answer)
            else:
                print(f"Failed to process {pdf_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"PDF file {pdf_path} not found or invalid: {str(e)}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")


def Test_Conversational_RAG(pdf_path, fine_tune=False):
    try:
        # Normalize path for Windows
        pdf_path = os.path.normpath(pdf_path)
        with open(pdf_path, "rb") as file:
            collection, processed = add_documents(file, os.path.basename(pdf_path))
            if collection:
                print(f"Processed: {processed}, Documents in collection: {collection.count()}")
                # Initialize Conversational RAG
                rag = ConversationalRAG(max_history=5)
                # Test with a multi-turn conversation
                queries = [
                    "Punishment of bigamy",
                    "Contents and verification of petitions",
                    "Permanent alimony and maintenance"
                ]
                for query in queries:
                    print(f"\nQuery: {query}")
                    answer = rag.conversational_rag(query, [collection], top_k=1, fine_tune=fine_tune)
                    print(answer)
            else:
                print(f"Failed to process {pdf_path}")
    except (FileNotFoundError, OSError) as e:
        print(f"PDF file {pdf_path} not found or invalid: {str(e)}")
    except Exception as e:
        print(f"Error processing {pdf_path}: {str(e)}")

def TestDirectory():
    # Test directory
    documents, ids, collections = load_pdfs_from_directory("pdfs")
    print(f"Loaded {len(documents)} chunks across {len(collections)} collections")
    results = query_collections("test query", collections, top_k=3)
    for result in results:
        print(f"Document: {result['document'][:50]}...")
        print(f"Metadata: {result['metadata']}")



if __name__ == "__main__":
    main()