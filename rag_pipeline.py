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
    def __init__(self, csv_path: str, embedding_model: str = "BAAI/bge-large-en-v1.5", few_shot = True, use_rag = True):
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA is not available! Please check your GPU setup.")
            sys.exit(1)
            
        print(f"Using GPU: {torch.cuda.get_device_name()}")
        
        self.use_rag = use_rag
        self.few_shot = few_shot
        
        if self.use_rag:
            # Initialize embeddings model
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cuda'}
            )
            
            # Load and process data
            self.df = pd.read_csv(csv_path)
            # Convert all values to strings and handle NaN values
            self.documents = self.df['text'].fillna('').astype(str).tolist()

            # Load cached embeddings if available
            if False:
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

        # Create different prompt templates and chains based on use_rag
        if self.use_rag:
            prompt_template = """[INST] Use the following pieces of context to answer the question at the end. 
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
""" + ("""
            Here are some examples of good question/answer pairs to follow:

            Question: Who is Pittsburgh named after?
            Answer: William Pitt

            Question: What famous machine learning venue had its first conference in Pittsburgh in 1980?
            Answer: ICML

            Question: What musical artist is performing at PPG Arena on October 13?
            Answer: Billie Eilish
""" if few_shot else "") + """
            Context: {context}

            Question: {question} [/INST]
            
            Answer: """
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain with RAG
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),
                chain_type_kwargs={"prompt": PROMPT},
                return_source_documents=True
            )
        else:
            # Simple prompt template without context for non-RAG mode
            prompt_template = """[INST] Answer the following question to the best of your ability.
""" + ("""
            Here are some examples of good question/answer pairs to follow:

            Question: Who is Pittsburgh named after?
            Answer: William Pitt

            Question: What famous machine learning venue had its first conference in Pittsburgh in 1980?
            Answer: ICML

            Question: What musical artist is performing at PPG Arena on October 13?
            Answer: Billie Eilish
""" if few_shot else "") + """
            Question: {question} [/INST]
            
            Answer: """
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["question"]
            )
            
            from langchain.chains import LLMChain
            # Create simple LLM chain without RAG
            self.qa_chain = LLMChain(
                llm=self.llm,
                prompt=PROMPT
            )
    
    def ask(self, question: str) -> dict:
        """Ask a question and get an answer based on the retrieved context."""
        if self.use_rag:
            # Get the result and context for RAG mode
            result = self.qa_chain.invoke({"query": question})
            
            # Print the answer and the context used
            print("\nAnswer:", result["result"])
            print("\nContext used for this answer:")
            source_texts = []
            for doc in result["source_documents"]:
                print("\n---")
                print(doc.page_content)
                source_texts.append(doc.page_content)
            
            # Pad source_texts to always have 5 elements (since k=5)
            source_texts.extend([""] * (5 - len(source_texts)))
            
            return {
                "answer": result["result"],
                "source_1": source_texts[0],
                "source_2": source_texts[1],
                "source_3": source_texts[2],
                "source_4": source_texts[3],
                "source_5": source_texts[4]
            }
        else:
            # Simple question-answering without context
            result = self.qa_chain.invoke({"question": question})
            print("\nAnswer:", result["text"])
            return {
                "answer": result["text"],
                "source_1": "",
                "source_2": "",
                "source_3": "",
                "source_4": "",
                "source_5": ""
            }

    def cleanup(self):
        """Clean up resources to free memory"""
        if hasattr(self, 'embeddings'):
            del self.embeddings
        if hasattr(self, 'vectorstore'):
            del self.vectorstore
        if hasattr(self, 'llm'):
            del self.llm
        if hasattr(self, 'qa_chain'):
            del self.qa_chain
        torch.cuda.empty_cache()

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


def batch_ask(pipeline: RAGPipeline, questions: list[str]):
    results = []
    failures = []
    sources_1 = []
    sources_2 = []
    sources_3 = []
    sources_4 = []
    sources_5 = []
    
    for i, question in enumerate(tqdm(questions, desc="Processing questions")):
        try:
            result = pipeline.ask(question)
            results.append(result["answer"])
            sources_1.append(result["source_1"])
            sources_2.append(result["source_2"])
            sources_3.append(result["source_3"])
            sources_4.append(result["source_4"])
            sources_5.append(result["source_5"])
        except Exception as e:
            print(f"\nFailed to process question {i}: {str(e)}")
            results.append("ERROR: Failed to process question")
            sources_1.append("")
            sources_2.append("")
            sources_3.append("")
            sources_4.append("")
            sources_5.append("")
            failures.append({
                "question_index": i,
                "question": question,
                "error": str(e)
            })
    
    # Log failures to a file
    if failures:
        failure_df = pd.DataFrame(failures)
        output_prefix = "baseline" if not pipeline.use_rag else (
            "rag_few_shot" if pipeline.few_shot else "rag"
        )
        failure_file = f"failures_{output_prefix}.csv"
        failure_df.to_csv(failure_file, index=False)
        print(f"\nLogged {len(failures)} failures to {failure_file}")
    
    return results, sources_1, sources_2, sources_3, sources_4, sources_5

def batch_ask_and_save_csv(pipeline: RAGPipeline, questions: list[str], output_file: str):
    results, sources_1, sources_2, sources_3, sources_4, sources_5 = batch_ask(pipeline, questions)
    df = pd.DataFrame({
        "question": questions, 
        "answer": results,
        "source_1": sources_1,
        "source_2": sources_2,
        "source_3": sources_3,
        "source_4": sources_4,
        "source_5": sources_5
    })
    df.to_csv(output_file, index=False)
    pipeline.cleanup()  # Clean up after we're done with this pipeline


if __name__ == "__main__":
    database = "all_combined_updated.csv"
    questions = pd.read_csv("gemma3_questions_balanced_sample_200.csv")["gemma3:12b_question"].tolist()

    good_embedding_model = "BAAI/bge-large-en-v1.5"
    bad_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"

    # Run RAG with bad embeddings
    print("\nRunning RAG pipeline with bad embeddings...")
    pipeline_rag_bad_embedding = RAGPipeline(database, use_rag=True, embedding_model=bad_embedding_model)
    batch_ask_and_save_csv(pipeline_rag_bad_embedding, questions, "results_rag_bad_embedding.csv")
    del pipeline_rag_bad_embedding
    torch.cuda.empty_cache()
    
    # Run baseline (no RAG)
    print("\nRunning baseline pipeline...")
    pipeline_baseline = RAGPipeline(database, use_rag=False, embedding_model=good_embedding_model)
    batch_ask_and_save_csv(pipeline_baseline, questions, "results_baseline.csv")
    del pipeline_baseline
    torch.cuda.empty_cache()

    # Run RAG with few-shot and good embeddings
    print("\nRunning RAG pipeline with few-shot and good embeddings...")
    pipeline_rag_few_shot = RAGPipeline(database, use_rag=True, embedding_model=good_embedding_model, few_shot=True)
    batch_ask_and_save_csv(pipeline_rag_few_shot, questions, "results_rag_few_shot.csv")
    del pipeline_rag_few_shot
    torch.cuda.empty_cache()

    # Run RAG with good embeddings (no few-shot)
    print("\nRunning RAG pipeline with good embeddings...")
    pipeline_rag = RAGPipeline(database, use_rag=True, embedding_model=good_embedding_model, few_shot=False)
    batch_ask_and_save_csv(pipeline_rag, questions, "results_rag.csv")
    del pipeline_rag
    torch.cuda.empty_cache()



    # Run RAG with few-shot and bad embeddings
    print("\nRunning RAG pipeline with few-shot and bad embeddings...")
    pipeline_rag_few_shot_bad_embedding = RAGPipeline(database, use_rag=True, embedding_model=bad_embedding_model, few_shot=True)
    batch_ask_and_save_csv(pipeline_rag_few_shot_bad_embedding, questions, "results_rag_few_shot_bad_embedding.csv")
    del pipeline_rag_few_shot_bad_embedding
    torch.cuda.empty_cache()

    print("\nAll pipelines completed successfully!")

