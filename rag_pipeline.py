import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.llms import LlamaCpp
import os
from dotenv import load_dotenv
import torch
import subprocess
import sys
from tqdm import tqdm
import time
import pickle

EMBEDDINGS_CACHE_PATH = "embeddings_cache.pkl"
EMBEDDINGS_HASH_PATH = "embeddings_hash.pkl"

def download_model():
    """Download the Mistral-7B-Instruct-v0.2.Q4_K_M model if not present"""
    model_path = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
    if not os.path.exists(model_path):
        print("Downloading Mistral-7B model...")
        subprocess.run([
            "wget", 
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        ], check=True)
    return model_path

class RAGPipeline:
    def __init__(self, csv_path: str):
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA is not available! Please check your GPU setup.")
            sys.exit(1)
        
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={'device': 'cuda'}
        )
        
        # Load and process data
        self.df = pd.read_csv(csv_path)
        # Convert all values to strings and handle NaN values
        # Convert all values to strings and handle NaN values
        self.documents = self.df['text'].fillna('').astype(str).tolist()

        # Load cached embeddings if available
        if os.path.exists(EMBEDDINGS_CACHE_PATH):
            with open(EMBEDDINGS_CACHE_PATH, 'rb') as f:
                all_embeddings = pickle.load(f)
        else:
            # Create vector store with progress tracking
            print("Generating embeddings. This may take some time...")
            
            # Batch processing with progress bar
            batch_size = 1000  # Adjust based on your GPU memory
            all_embeddings = []
            
            for i in tqdm(range(0, len(self.documents), batch_size), desc="Generating embeddings"):
                batch = self.documents[i:i+batch_size]
                all_embeddings.extend(self.embeddings.embed_documents(batch))
            
            # Save embeddings and hash
            with open(EMBEDDINGS_CACHE_PATH, 'wb') as f:
                pickle.dump(all_embeddings, f)
        
        # Create FAISS index from the embeddings
        print("Building FAISS index...")
        text_embedding_pairs = list(zip(self.documents, all_embeddings))
        self.vectorstore = FAISS.from_embeddings(
            text_embeddings=text_embedding_pairs,
            embedding=self.embeddings
        )
        
        # Download and initialize the local LLM
        model_path = download_model()
        
        # Initialize LLM with CUDA support
        self.llm = LlamaCpp(
            model_path=model_path,
            n_gpu_layers=-1,  # Use all layers on GPU
            n_ctx=4096,       # Context window
            n_batch=512,      # Batch size for prompt processing
            temperature=0.1,
            max_tokens=2048,
            top_p=0.95,
            verbose=True,
            n_threads=8,      # Number of CPU threads
            use_mlock=True,   # Lock memory to prevent swapping
            use_mmap=True,    # Use memory mapping for faster loading
        )

        
        # Create prompt template
        prompt_template = """[INST] Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Here are some examples of good question/answer pairs to follow:

        Question: Who is the mayor of Pittsburgh?
        Answer: Ed Gainey

        Question: Where do the Pirates play?
        Answer: PNC Park

        Question: What neighborhood is the University of Pittsburgh in?
        Answer: Oakland

        Context: {context}

        Question: {question} [/INST]
        
        Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # Create QA chain
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ask(self, question: str) -> str:
        """Ask a question and get an answer based on the retrieved context."""
        # Get the result and context
        result = self.qa_chain.invoke({"query": question})
        
        # Print the answer and the context used
        print("\nAnswer:", result["result"])
        print("\nContext used for this answer:")
        for doc in result["source_documents"]:
            print("\n---")
            print(doc.page_content)
        
        return result["result"]

def main():
    # Initialize the pipeline
    print("Initializing RAG Pipeline...")
    pipeline = RAGPipeline("all_combined.csv")
    
    # Interactive question answering loop
    print("\nRAG Pipeline initialized! Ask questions (type 'quit' to exit):")
    while True:
        question = input("\nQuestion: ")
        if question.lower() == 'quit':
            break
        
        try:
            pipeline.ask(question)
        except Exception as e:
            print(f"\nError occurred: {str(e)}")

if __name__ == "__main__":
    main() 