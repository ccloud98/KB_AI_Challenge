import os
import json
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# --- CONFIG (기존 코드와 동일) ---
PDF_FOLDER = "KB_web_docs"
VECTOR_STORE_DIR = "vector_store"
os.environ["GOOGLE_API_KEY"] = "AIzaSyAc_-QEO_4M7u7carnU3LYbX626OvNuRW8"

def analyze_vector_store():
    """Analyze vector store structure and content"""
    print("="*80)
    print("VECTOR STORE ANALYSIS")
    print("="*80)
    
    # Check if vector store exists
    if not os.path.exists(VECTOR_STORE_DIR):
        print("Vector store directory does not exist!")
        print(f"Expected location: {VECTOR_STORE_DIR}")
        return
    
    # Check directory contents
    print(f"Vector store directory: {VECTOR_STORE_DIR}")
    print(f"Directory contents: {os.listdir(VECTOR_STORE_DIR)}")
    print()
    
    try:
        # Initialize embeddings and load vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        
        # Get all data
        all_data = db.get()
        
        print("BASIC STATISTICS")
        print("-" * 40)
        print(f"Total documents (chunks): {len(all_data['documents'])}")
        print(f"Total metadata entries: {len(all_data['metadatas'])}")
        print(f"Total IDs: {len(all_data['ids'])}")
        print()
        
        # Analyze by source files
        sources = {}
        for i, metadata in enumerate(all_data['metadatas']):
            source = metadata.get('source', 'Unknown')
            if source not in sources:
                sources[source] = {
                    'count': 0,
                    'sample_ids': [],
                    'sample_content': [],
                    'all_metadata': []
                }
            sources[source]['count'] += 1
            sources[source]['sample_ids'].append(all_data['ids'][i])
            sources[source]['sample_content'].append(all_data['documents'][i][:100])
            sources[source]['all_metadata'].append(metadata)
        
        print("SOURCE FILES ANALYSIS")
        print("-" * 40)
        for source, info in sources.items():
            print(f"File: {source}")
            print(f"  Chunks: {info['count']}")
            print(f"  Sample ID: {info['sample_ids'][0]}")
            print(f"  Content preview: {info['sample_content'][0]}...")
            print()
        
        return all_data, sources
        
    except Exception as e:
        print(f"Error accessing vector store: {e}")
        return None, None

def analyze_document_content(all_data, sources):
    """Analyze document content in detail"""
    if not all_data:
        return
    
    print("="*80)
    print("DOCUMENT CONTENT ANALYSIS")
    print("="*80)
    
    # Sample documents analysis
    print("SAMPLE DOCUMENTS (First 3 chunks)")
    print("-" * 40)
    for i in range(min(3, len(all_data['documents']))):
        print(f"Chunk {i+1}:")
        print(f"  ID: {all_data['ids'][i]}")
        print(f"  Metadata: {all_data['metadatas'][i]}")
        print(f"  Content length: {len(all_data['documents'][i])} characters")
        print(f"  Content preview: {all_data['documents'][i][:200]}...")
        print("-" * 40)
    
    # Content length statistics
    lengths = [len(doc) for doc in all_data['documents']]
    print("\nCONTENT LENGTH STATISTICS")
    print("-" * 40)
    print(f"Average chunk length: {sum(lengths)/len(lengths):.1f} characters")
    print(f"Minimum chunk length: {min(lengths)} characters")
    print(f"Maximum chunk length: {max(lengths)} characters")
    print()
    
    # Unique metadata fields
    all_metadata_keys = set()
    for metadata in all_data['metadatas']:
        all_metadata_keys.update(metadata.keys())
    
    print("METADATA FIELDS FOUND")
    print("-" * 40)
    for key in sorted(all_metadata_keys):
        sample_values = set()
        for metadata in all_data['metadatas'][:10]:  # Sample first 10
            if key in metadata:
                sample_values.add(str(metadata[key]))
        print(f"  {key}: {list(sample_values)[:3]}...")  # Show first 3 unique values
    print()

def search_content_samples(keyword="개인정보"):
    """Search for specific content samples"""
    print("="*80)
    print(f"CONTENT SEARCH: '{keyword}'")
    print("="*80)
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        db = Chroma(persist_directory=VECTOR_STORE_DIR, embedding_function=embeddings)
        
        # Search for relevant chunks
        results = db.similarity_search(keyword, k=5)
        
        print(f"Found {len(results)} relevant chunks:")
        print("-" * 40)
        
        for i, doc in enumerate(results, 1):
            print(f"Result {i}:")
            print(f"  Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"  Content: {doc.page_content[:300]}...")
            print("-" * 40)
        
        return results
        
    except Exception as e:
        print(f"Search failed: {e}")
        return None

def check_pdf_source_files():
    """Check original PDF files"""
    print("="*80)
    print("ORIGINAL PDF FILES ANALYSIS")
    print("="*80)
    
    if not os.path.exists(PDF_FOLDER):
        print(f"PDF folder does not exist: {PDF_FOLDER}")
        return
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]
    
    print(f"PDF folder: {PDF_FOLDER}")
    print(f"Total PDF files: {len(pdf_files)}")
    print("-" * 40)
    
    for i, filename in enumerate(pdf_files, 1):
        filepath = os.path.join(PDF_FOLDER, filename)
        file_size = os.path.getsize(filepath)
        print(f"{i}. {filename}")
        print(f"   Size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    print()

def full_analysis():
    """Run complete analysis"""
    print("STARTING COMPREHENSIVE DOCUMENT ANALYSIS")
    print("="*80)
    
    # 1. Check original PDF files
    check_pdf_source_files()
    
    # 2. Analyze vector store
    all_data, sources = analyze_vector_store()
    
    # 3. Analyze document content
    if all_data:
        analyze_document_content(all_data, sources)
    
    # 4. Search samples
    search_content_samples("개인정보")
    search_content_samples("privacy")
    
    print("="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)

if __name__ == "__main__":
    full_analysis()