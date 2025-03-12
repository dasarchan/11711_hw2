import pandas as pd
from pathlib import Path
import os

# LlamaIndex imports
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import Document
import chromadb

class LlamaIndexRAG:
    def __init__(self, csv_path, model_path, embeddings_model_name, persist_directory):
        self.csv_path = csv_path
        self.model_path = model_path
        self.embeddings_model_name = embeddings_model_name
        self.persist_directory = persist_directory
        
        # Initialize components
        self.documents = None
        self.llm = None
        self.db = None
        self.vector_store = None
        self.index = None
        self.query_engine = None
        
    def load_data(self):
        """Load pre-chunked CSV data with source and text columns"""
        print("Loading pre-chunked CSV data...")
        try:
            # Load CSV file with progress indication
            print("Reading CSV file...")
            df = pd.read_csv(self.csv_path)
            print(f"✓ CSV loaded with {len(df)} rows")
            
            # Check for required columns
            if 'source' not in df.columns or 'text' not in df.columns:
                print("Error: CSV must contain 'source' and 'text' columns")
                return False
            
            # Convert each row to a document with progress updates
            self.documents = []
            skipped_rows = 0
            total_rows = len(df)
            
            print("Converting rows to documents...")
            update_interval = max(1, total_rows // 20)  # Update progress every 5%
            
            for i, (_, row) in enumerate(df.iterrows()):
                # Display progress periodically
                if i % update_interval == 0:
                    progress = (i / total_rows) * 100
                    print(f"Progress: {progress:.1f}% ({i}/{total_rows})")
                
                # Skip rows with NaN in text column
                if pd.isna(row['text']):
                    skipped_rows += 1
                    continue
                    
                # Handle potential NaN in source column
                source = row['source'] if not pd.isna(row['source']) else "unknown"
                
                doc = Document(
                    text=str(row['text']),  # Ensure text is converted to string
                    metadata={'source': source}
                )
                self.documents.append(doc)
            
            print(f"✓ Loaded {len(self.documents)} pre-chunked documents")
            if skipped_rows > 0:
                print(f"⚠ Skipped {skipped_rows} rows with empty or NaN text values")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def setup_llm(self):
        """Initialize the LLM (Llama2)"""
        print("Setting up LLM...")
        try:
            self.llm = LlamaCPP(
                model_path=self.model_path,
                temperature=0.1,
                max_new_tokens=100,  # Use max_new_tokens instead of max_tokens
                context_window=2048
            )
            print(f"Initialized Llama2 model from {self.model_path}")
            return True
        except Exception as e:
            print(f"Error loading LLM: {e}")
            return False
    
    def setup_embeddings(self):
        """Setup embedding model"""
        print("Setting up embeddings model...")
        try:
            embed_model = HuggingFaceEmbedding(
                model_name=self.embeddings_model_name
            )
            Settings.embed_model = embed_model
            print(f"Initialized embeddings model: {self.embeddings_model_name}")
            return True
        except Exception as e:
            print(f"Error setting up embeddings: {e}")
            return False
    
    def setup_vectorstore(self):
        """Setup Chroma vector store"""
        print("Setting up vector database...")
        try:
            # Create Chroma client
            chroma_client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Create or get collection
            chroma_collection = chroma_client.get_or_create_collection("rag_collection")
            
            # Create vector store
            self.vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            
            # Check if collection already has documents
            if chroma_collection.count() > 0:
                print(f"✓ Using existing Chroma collection with {chroma_collection.count()} documents")
            else:
                print("Creating new vector store index...")
                
                # Create storage context
                storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
                
                # Create index from documents with progress updates
                print(f"Embedding {len(self.documents)} documents...")
                
                # Create batches for embedding with progress reporting
                batch_size = 100
                total_batches = (len(self.documents) + batch_size - 1) // batch_size
                
                for i in range(0, total_batches):
                    start_idx = i * batch_size
                    end_idx = min((i + 1) * batch_size, len(self.documents))
                    batch = self.documents[start_idx:end_idx]
                    
                    # Print progress
                    progress = (i / total_batches) * 100
                    print(f"Embedding progress: {progress:.1f}% (batch {i+1}/{total_batches})")
                    
                    # For the last batch, create the index with all documents
                    if i == total_batches - 1:
                        self.index = VectorStoreIndex.from_documents(
                            self.documents,
                            storage_context=storage_context
                        )
                
                print(f"✓ Added {len(self.documents)} documents to vector store")
            
            return True
        except Exception as e:
            print(f"Error setting up vector store: {e}")
            return False
    
    def setup_query_engine(self):
        """Setup the query engine for RAG"""
        print("Setting up query engine...")
        try:
            # If index not already created, create it from the vector store
            if self.index is None:
                self.index = VectorStoreIndex.from_vector_store(self.vector_store)
            
            # Set up the query engine
            Settings.llm = self.llm
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=2,
                streaming=True
            )
            
            print("Query engine ready")
            return True
        except Exception as e:
            print(f"Error setting up query engine: {e}")
            return False
    
    def initialize(self):
        """Initialize the complete RAG pipeline"""
        if not self.load_data():
            return False
        
        if not self.setup_embeddings():
            return False
        
        if not self.setup_llm():
            return False
        
        if not self.setup_vectorstore():
            return False
        
        if not self.setup_query_engine():
            return False
        
        print("LlamaIndex RAG system initialization complete!")
        return True
    
    def ask_question(self, question):
        """Process a question through the RAG pipeline"""
        if self.query_engine is None:
            print("Query engine not initialized. Please run initialize() first.")
            return None
        
        print(f"\nQuestion: {question}")
        print("Retrieving information...")
        
        try:
            # We need to handle streaming differently
            if hasattr(self.query_engine, "streaming") and self.query_engine.streaming:
                print("\nAnswer:")
                # For streaming, we need to iterate through the response stream
                response_stream = self.query_engine.query(question)
                # Collect the full response for later use
                full_response_text = ""
                for token in response_stream:
                    print(token, end="", flush=True)
                    full_response_text += token
                
                print("\n")
                response = full_response_text
            else:
                # Non-streaming response
                response = self.query_engine.query(question)
                print("\nAnswer:")
                response.print_response_stream()
                # print(response.response)
                
                # Get source nodes if available
                if hasattr(response, 'source_nodes'):
                    print("\nRetrieved sources:")
                    for i, node in enumerate(response.source_nodes):
                        print(f"Source {i+1}:")
                        print(f"  Content: {node.text[:150]}...")
                        if i < len(response.source_nodes) - 1:
                            print("  -" * 30)
            
            return response
        except Exception as e:
            print(f"\nError processing question: {e}")
            import traceback
            traceback.print_exc()
            return None

def main():
    # Configuration
    MODEL_PATH = "models/tinyllama-1.1b-chat-v1.0.Q2_K.gguf"  # Path to your Llama2 model
    EMBEDDINGS_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    CSV_FILE_PATH = "all_combined.csv"  # Path to your CSV file with source and text columns
    PERSIST_DIRECTORY = "db"
    
    # Create and initialize the RAG system
    rag = LlamaIndexRAG(
        csv_path=CSV_FILE_PATH,
        model_path=MODEL_PATH,
        embeddings_model_name=EMBEDDINGS_MODEL_NAME,
        persist_directory=PERSIST_DIRECTORY
    )
    
    if not rag.initialize():
        print("Failed to initialize RAG system.")
        return
    
    # Interactive question answering loop
    print("\n" + "="*50)
    print("LlamaIndex RAG Question Answering System Ready")
    print("="*50)
    print("Type 'exit' to quit the program.")
    
    while True:
        question = input("\nEnter your question: ")
        
        if question.lower() in ["exit", "quit", "q"]:
            break
        
        rag.ask_question(question)

if __name__ == "__main__":
    from huggingface_hub import hf_hub_download
    hf_hub_download(repo_id="TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF", filename="tinyllama-1.1b-chat-v1.0.Q2_K.gguf", local_dir="models")
    main()