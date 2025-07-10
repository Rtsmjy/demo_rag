import os
import torch
import gradio as gr
from typing import List, Tuple, Dict, Generator
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.schema import Document
import requests
import json
from pathlib import Path
import shutil
import time
import threading

# GPU configuration and optimization
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
EMBEDDING_MODEL = "mxbai-embed-large:latest"
LLM_MODEL = "qwen3:30b"
RERANK_MODEL = "hf.co/Mungert/Qwen3-Reranker-4B-GGUF:Q4_K_M"
CHROMA_PERSIST_DIR = "./chroma_db"
UPLOAD_DIR = "./uploaded_docs"
BATCH_SIZE = 32

# Create necessary directories
os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

class OllamaEmbeddingFunction:
    """Optimized Ollama Embedding Function"""
    def __init__(self, model_name: str, base_url: str = OLLAMA_BASE_URL):
        self.model_name = model_name
        self.base_url = base_url
        self.session = requests.Session()
    
    def name(self) -> str:
        """Return the name of the embedding function"""
        return f"ollama_{self.model_name}"
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        """Batch process embeddings"""
        embeddings = []
        
        for i in range(0, len(input), BATCH_SIZE):
            batch = input[i:i + BATCH_SIZE]
            batch_embeddings = self._get_batch_embeddings(batch)
            embeddings.extend(batch_embeddings)
        
        return embeddings
    
    def _get_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get batch embeddings"""
        try:
            response = self.session.post(
                f"{self.base_url}/api/embed",
                json={"model": self.model_name, "input": texts},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()["embeddings"]
            else:
                return [self._get_single_embedding(text) for text in texts]
        except:
            return [self._get_single_embedding(text) for text in texts]
    
    def _get_single_embedding(self, text: str) -> List[float]:
        """Get single embedding"""
        response = self.session.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model_name, "input": text}
        )
        if response.status_code == 200:
            return response.json()["embeddings"][0]
        else:
            raise Exception(f"Embedding API error: {response.text}")

# Initialize ChromaDB
embedding_function = OllamaEmbeddingFunction(EMBEDDING_MODEL)
client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

# Get or create collection
collection_name = "rag_collection"
try:
    collection = client.get_collection(
        name=collection_name,
        embedding_function=embedding_function
    )
    print(f"Loaded existing collection: {collection_name}")
except:
    collection = client.create_collection(
        name=collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created new collection: {collection_name}")

def call_ollama_llm_stream(prompt: str, model: str = LLM_MODEL) -> Generator[str, None, None]:
    """Stream call to Ollama LLM"""
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_ctx": 8192,
                    "num_gpu": -1,
                }
            },
            stream=True,
            timeout=120
        )
        
        if response.status_code == 200:
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if 'response' in data:
                            yield data['response']
                        if data.get('done', False):
                            break
                    except json.JSONDecodeError:
                        continue
        else:
            yield f"Error: {response.text}"
    except Exception as e:
        yield f"Error: {str(e)}"

def call_ollama_llm(prompt: str, model: str = LLM_MODEL) -> str:
    """Non-stream call to Ollama LLM"""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": 8192,
                "num_gpu": -1,
            }
        }
    )
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"Error: {response.text}"

def check_gpu_memory():
    """Check GPU memory status"""
    if not torch.cuda.is_available():
        return True, "No GPU available"
    
    try:
        torch.cuda.synchronize()
        memory_allocated = torch.cuda.memory_allocated() / 1024**3
        memory_reserved = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        
        free_memory = total_memory - memory_reserved
        memory_usage = memory_reserved / total_memory
        
        print(f"🔍 GPU Memory Status: Used {memory_reserved:.2f}GB / {total_memory:.2f}GB ({memory_usage:.1%})")
        
        # If memory usage exceeds 80%, suggest cleanup
        if memory_usage > 0.8:
            torch.cuda.empty_cache()
            print("🧹 Cleaning GPU cache...")
            
        return memory_usage < 0.9, f"Memory usage: {memory_usage:.1%}"
    except Exception as e:
        return False, f"GPU check failed: {str(e)}"

def process_document(file_path: str, use_mineru: bool = False) -> List[Document]:
    """Process document with optional MinerU (GPU mode)"""
    documents = []
    
    if use_mineru and file_path.endswith('.pdf'):
        try:
            import subprocess
            output_dir = f"{UPLOAD_DIR}/mineru_output"
            os.makedirs(output_dir, exist_ok=True)
            
            # Check GPU memory status
            gpu_ok, gpu_msg = check_gpu_memory()
            if not gpu_ok:
                print(f"⚠️ GPU memory insufficient, switching to CPU mode: {gpu_msg}")
                cmd = f'mineru pdf2md -p "{file_path}" -o {output_dir} -d cpu'
                print(f"🔄 Running MinerU (CPU mode): {cmd}")
                device_mode = "MinerU-CPU-Fallback"
            else:
                # Use GPU mode with resource management
                print(f"🚀 GPU memory sufficient, using GPU mode: {gpu_msg}")
                cmd = f'mineru pdf2md -p "{file_path}" -o {output_dir} -d cuda'
                print(f"🔄 Running MinerU (GPU mode): {cmd}")
                print("📊 MinerU processing progress will be displayed in terminal...")
                device_mode = "MinerU-GPU"
            
            # Set process priority and resource limits
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = '0'  # Limit to single GPU
            env['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'  # Limit memory allocation
            
            # Execute MinerU processing - output to terminal
            print(f"🔄 Starting MinerU processing, please observe terminal output...")
            result = subprocess.run(
                cmd, 
                shell=True, 
                capture_output=False,  # Don't capture output, let it display directly to terminal
                text=True, 
                # timeout=1200,  # Increase timeout
                env=env
            )
            
            # Check execution result
            if result.returncode != 0:
                print(f"⚠️ MinerU execution failed (return code: {result.returncode})")
                print("🔄 GPU mode failed, retrying with CPU mode...")
                cmd_cpu = f'mineru pdf2md -p "{file_path}" -o {output_dir} -d cpu'
                print(f"🔄 Running MinerU (CPU retry): {cmd_cpu}")
                result = subprocess.run(cmd_cpu, shell=True, capture_output=False, text=True, timeout=900)
                device_mode = "MinerU-CPU-Retry"
                    
                if result.returncode != 0:
                    raise Exception(f"MinerU failed even with CPU mode (return code: {result.returncode})")
            
            # Check if output files were generated
            md_files = list(Path(output_dir).glob("**/*.md"))
            
            if md_files:
                for md_file in md_files:
                    with open(md_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if len(content.strip()) > 0:  # Ensure content is not empty
                            if len(content) > 5000:
                                chunks = [content[i:i+5000] for i in range(0, len(content), 4500)]
                                for chunk in chunks:
                                    documents.append(Document(
                                        page_content=chunk, 
                                        metadata={"source": file_path, "processed_by": device_mode}
                                    ))
                            else:
                                documents.append(Document(
                                    page_content=content, 
                                    metadata={"source": file_path, "processed_by": device_mode}
                                ))
                print(f"✅ Successfully processed {file_path} with {device_mode} ({len(md_files)} files, {len(documents)} segments)")
            else:
                print(f"⚠️ MinerU completed but no output files found")
                raise Exception("No MinerU output files")
                
            # Clean GPU cache after processing
            if device_mode.startswith("MinerU-GPU") and torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🧹 MinerU processing complete, cleaning GPU cache")
                
        except Exception as e:
            print(f"MinerU error: {e}, falling back to PyMuPDF")
            # Ensure GPU cache cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            loader = PyMuPDFLoader(file_path)
            documents = loader.load()
    elif file_path.endswith('.pdf'):
        loader = PyMuPDFLoader(file_path)
        documents = loader.load()
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
    elif file_path.endswith('.md'):
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
    else:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents = [Document(page_content=content, metadata={"source": file_path})]
        except:
            print(f"Unsupported file type: {file_path}")
    
    return documents

def add_documents_to_collection(documents: List[Document]):
    """Add documents to vector database"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
    )
    
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        return
    
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    existing_ids = collection.get()["ids"]
    start_id = len(existing_ids)
    ids = [f"doc_{start_id + i}" for i in range(len(chunks))]
    
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_metadatas = metadatas[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        
        collection.add(
            documents=batch_texts,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
    
    print(f"Added {len(texts)} chunks to collection")

def query_knowledge_base_stream(query: str, top_k: int = 5) -> Generator[Tuple[str, List[str]], None, None]:
    """Stream query knowledge base and generate answers"""
    # Show search status first
    yield "🔍 Searching relevant documents...", []
    
    # Search relevant documents
    results = collection.query(
        query_texts=[query],
        n_results=min(top_k, len(collection.get()["ids"])),
    )
    
    if not results["documents"][0]:
        yield "❌ No relevant documents found.", []
        return
    
    # Build context
    documents = results["documents"][0][:top_k]
    metadatas = results["metadatas"][0][:top_k]
    sources = list(set([m.get("source", "Unknown") for m in metadatas]))
    
    yield f"📚 Found {len(documents)} relevant document segments", sources
    
    context = "\n\n---\n\n".join(documents)
    
    # Build prompt
    prompt = f"""You are a professional Q&A assistant. Please answer user questions accurately based on the provided context information.

Requirements:
1. If the context contains the answer, please provide a detailed and accurate response
2. If information is insufficient, clearly state what information is missing
3. Stay objective and do not fabricate information
4. Appropriately cite original text from the context

Context Information:
{context}

User Question: {query}

Detailed Answer:"""
    
    # Show generation start status
    yield "🤖 AI is thinking and generating answer...\n", sources
    
    # Stream generate answer
    full_answer = "🤖 AI is thinking and generating answer...\n\n"
    for chunk in call_ollama_llm_stream(prompt):
        if chunk.strip():
            full_answer += chunk
            yield full_answer, sources
    
    # Ensure final answer doesn't contain status information
    final_answer = full_answer.replace("🤖 AI is thinking and generating answer...\n\n", "")
    yield final_answer, sources

# Create Gradio interface
with gr.Blocks(title="Streaming RAG System") as demo:
    gr.Markdown("# 🚀 Ollama-based Streaming RAG System")
    gr.Markdown(f"""
    **System Status:**
    - GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}
    - Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB
    - MinerU Mode: 🚀 GPU Smart Parsing (Enabled)
    - Embedding Model: {EMBEDDING_MODEL}
    - LLM Model: {LLM_MODEL}
    """)
    
    with gr.Tab("Document Management"):
        with gr.Row():
            with gr.Column():
                file_input = gr.File(label="Select Document", file_types=[".pdf", ".txt", ".md"])
                dir_input = gr.Textbox(label="Or enter directory path (batch processing)", placeholder="/path/to/documents")
                use_mineru_checkbox = gr.Checkbox(label="Use MinerU for complex PDF processing (GPU Smart Parsing)", value=False)
                
                with gr.Row():
                    upload_btn = gr.Button("Upload Document", variant="primary")
                    load_from_disk_btn = gr.Button("Load from Disk", variant="secondary")
                
            with gr.Column():
                upload_status = gr.Textbox(label="Processing Status", lines=8, interactive=False)
                collection_info = gr.Textbox(label="Knowledge Base Info", interactive=False)
                
                with gr.Row():
                    refresh_btn = gr.Button("Refresh Info")
                    clear_btn = gr.Button("Clear Knowledge Base", variant="stop")
                
                with gr.Row():
                    delete_docs_checkbox = gr.Checkbox(
                        label="Also delete uploaded original document files", 
                        value=False,
                        info="⚠️ This will delete all files in the uploaded_docs directory"
                    )
                    confirm_clear_btn = gr.Button("🗑️ Complete Clear & Delete", variant="stop")
    
    with gr.Tab("Smart Q&A 💬"):
        with gr.Row():
            with gr.Column():
                query_input = gr.Textbox(
                    label="Enter your question", 
                    placeholder="Please enter what you want to know...",
                    lines=3
                )
                
                top_k_slider = gr.Slider(minimum=1, maximum=20, value=5, step=1, label="Number of documents to retrieve")
                
                with gr.Row():
                    query_btn = gr.Button("🚀 Start Query", variant="primary")
                    stop_btn = gr.Button("⏹️ Stop Generation", variant="stop")
                
            with gr.Column():
                answer_output = gr.Textbox(label="AI Answer (Real-time Generation)", lines=15, interactive=False)
                sources_output = gr.Textbox(label="Information Sources", lines=3, interactive=False)
    
    # Define functions
    def get_collection_info():
        """Get knowledge base information"""
        try:
            data = collection.get()
            count = len(data["ids"])
            sources = set()
            for metadata in data["metadatas"]:
                if metadata and "source" in metadata:
                    sources.add(metadata["source"])
            
            info = f"📊 Knowledge Base Statistics:\n"
            info += f"📝 Document segments: {count}\n"
            info += f"📚 Document sources: {len(sources)}\n"
            if sources:
                info += "\n📁 Document sources:\n"
                for source in list(sources)[:10]:
                    info += f"  • {os.path.basename(source)}\n"
                if len(sources) > 10:
                    info += f"  ... and {len(sources) - 10} more documents"
            
            return info
        except:
            return "Knowledge base is empty"
    
    def upload_file_handler(file, use_mineru):
        """Handle file upload"""
        if file is None:
            return "⚠️ Please select a file", get_collection_info()
        
        file_name = os.path.basename(file.name)
        file_path = os.path.join(UPLOAD_DIR, file_name)
        
        try:
            shutil.copy(file.name, file_path)
            status = f"📁 File received: {file_name}\n"
            
            file_size = os.path.getsize(file_path)
            status += f"📊 File size: {file_size / 1024:.1f} KB\n"
            
            status += "🔄 Starting document processing...\n"
            documents = process_document(file_path, use_mineru)
            
            if documents and len(documents) > 0:
                total_chars = sum(len(doc.page_content) for doc in documents)
                status += f"📝 Extracted {len(documents)} document segments, {total_chars} characters total\n"
                
                mineru_used = any(d.metadata.get("processed_by", "").startswith("MinerU") for d in documents)
                if mineru_used:
                    status += "🚀 Processed with MinerU (Smart Parsing)\n"
                
                status += "🔄 Generating vectors and storing to knowledge base...\n"
                before_count = len(collection.get()["ids"])
                add_documents_to_collection(documents)
                after_count = len(collection.get()["ids"])
                new_chunks = after_count - before_count
                
                status += f"✅ Processing completed successfully!\n"
                status += f"💾 Added {new_chunks} knowledge segments to database\n"
                status += f"📚 Knowledge base total: {after_count} segments"
            else:
                status += "⚠️ Document is empty or content cannot be extracted\n"
                status += "Possible reasons: encrypted document, unsupported format, or corrupted file"
                
        except Exception as e:
            status = f"❌ Error processing document:\n"
            status += f"Error message: {str(e)}\n"
            status += f"File: {file_name}"
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
        
        return status, get_collection_info()
    
    def load_from_disk_handler(dir_path, use_mineru):
        """Load existing documents from disk"""
        if not dir_path:
            # If no path specified, use default upload directory
            dir_path = UPLOAD_DIR
        
        if not os.path.exists(dir_path):
            return f"❌ Path does not exist: {dir_path}", get_collection_info()
        
        supported_extensions = ['.pdf', '.txt', '.md']
        found_files = []
        
        for ext in supported_extensions:
            found_files.extend(Path(dir_path).glob(f"**/*{ext}"))
        
        if not found_files:
            return f"⚠️ No supported document files found in {dir_path}", get_collection_info()
        
        status = f"🔍 Found {len(found_files)} documents in {dir_path}\n\n"
        processed = 0
        failed = 0
        
        for file_path in found_files:
            try:
                status += f"🔄 Processing: {file_path.name}...\n"
                documents = process_document(str(file_path), use_mineru)
                if documents:
                    add_documents_to_collection(documents)
                    processed += 1
                    status += f"  ✅ Success: {len(documents)} segments\n"
                else:
                    failed += 1
                    status += f"  ⚠️ No content\n"
            except Exception as e:
                failed += 1
                status += f"  ❌ Failed: {str(e)}\n"
        
        status += f"\n📊 Processing complete: {processed} successful, {failed} failed"
        return status, get_collection_info()
    
    def clear_knowledge_base():
        """Completely clear knowledge base and delete all data"""
        global client, collection
        try:
            status_msg = "🗑️ Starting knowledge base cleanup...\n"
            
            # 1. Delete ChromaDB collection
            try:
                client.delete_collection(name=collection_name)
                status_msg += "✅ ChromaDB collection deleted\n"
            except Exception as e:
                status_msg += f"⚠️ Issue deleting collection: {str(e)}\n"
            
            # 2. Clean GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                status_msg += "✅ GPU cache cleaned\n"
            
            # 3. Completely delete persistent data directory
            if os.path.exists(CHROMA_PERSIST_DIR):
                try:
                    import shutil
                    shutil.rmtree(CHROMA_PERSIST_DIR)
                    status_msg += f"✅ Data directory deleted: {CHROMA_PERSIST_DIR}\n"
                except Exception as e:
                    status_msg += f"⚠️ Issue deleting data directory: {str(e)}\n"
            
            # 4. Recreate data directory
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            status_msg += "✅ Data directory recreated\n"
            
            # 5. Reinitialize ChromaDB client and collection
            try:
                # Recreate client
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                status_msg += "✅ ChromaDB client reinitialized\n"
                
                # Recreate collection
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                status_msg += "✅ Knowledge base collection recreated\n"
                
            except Exception as e:
                status_msg += f"❌ Error reinitializing database: {str(e)}\n"
                return status_msg, get_collection_info()
            
            # 6. Optional: clean uploaded documents (preserve original files, only clean processed files)
            try:
                mineru_output = f"{UPLOAD_DIR}/mineru_output"
                if os.path.exists(mineru_output):
                    shutil.rmtree(mineru_output)
                    status_msg += "✅ MinerU processing cache cleaned\n"
            except Exception as e:
                status_msg += f"⚠️ Issue cleaning processing cache: {str(e)}\n"
            
            status_msg += "\n🎉 Knowledge base completely cleared, all data deleted!"
            
            return status_msg, get_collection_info()
            
        except Exception as e:
            error_msg = f"❌ Serious error while clearing knowledge base:\n{str(e)}\n"
            error_msg += "📋 Suggest manually checking these directories:\n"
            error_msg += f"  • {CHROMA_PERSIST_DIR}\n"
            error_msg += f"  • {UPLOAD_DIR}/mineru_output\n"
            return error_msg, get_collection_info()
    
    def clear_knowledge_base_completely(delete_docs: bool = False):
        """Completely clear knowledge base, optionally delete original documents"""
        global client, collection
        try:
            status_msg = "🗑️ Starting complete knowledge base cleanup...\n"
            
            # 1. Delete ChromaDB collection
            try:
                client.delete_collection(name=collection_name)
                status_msg += "✅ ChromaDB collection deleted\n"
            except Exception as e:
                status_msg += f"⚠️ Issue deleting collection: {str(e)}\n"
            
            # 2. Clean GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                status_msg += "✅ GPU cache cleaned\n"
            
            # 3. Completely delete persistent data directory
            if os.path.exists(CHROMA_PERSIST_DIR):
                try:
                    import shutil
                    shutil.rmtree(CHROMA_PERSIST_DIR)
                    status_msg += f"✅ ChromaDB data directory deleted\n"
                except Exception as e:
                    status_msg += f"⚠️ Issue deleting data directory: {str(e)}\n"
            
            # 4. Delete original document files (if user chooses)
            if delete_docs and os.path.exists(UPLOAD_DIR):
                try:
                    # Count files to be deleted
                    file_count = 0
                    for root, dirs, files in os.walk(UPLOAD_DIR):
                        file_count += len(files)
                    
                    shutil.rmtree(UPLOAD_DIR)
                    status_msg += f"✅ Upload directory and {file_count} files deleted\n"
                except Exception as e:
                    status_msg += f"⚠️ Issue deleting upload directory: {str(e)}\n"
            else:
                # Only clean processing cache, keep original documents
                try:
                    mineru_output = f"{UPLOAD_DIR}/mineru_output"
                    if os.path.exists(mineru_output):
                        shutil.rmtree(mineru_output)
                        status_msg += "✅ MinerU processing cache cleaned (original documents preserved)\n"
                except Exception as e:
                    status_msg += f"⚠️ Issue cleaning processing cache: {str(e)}\n"
            
            # 5. Recreate necessary directories
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            os.makedirs(UPLOAD_DIR, exist_ok=True)
            status_msg += "✅ Necessary directories recreated\n"
            
            # 6. Reinitialize ChromaDB client and collection
            try:
                # Recreate client
                client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
                status_msg += "✅ ChromaDB client reinitialized\n"
                
                # Recreate collection
                collection = client.create_collection(
                    name=collection_name,
                    embedding_function=embedding_function,
                    metadata={"hnsw:space": "cosine"}
                )
                status_msg += "✅ Knowledge base collection recreated\n"
                
            except Exception as e:
                status_msg += f"❌ Error reinitializing database: {str(e)}\n"
                return status_msg, get_collection_info()
            
            # 7. Display cleanup results
            if delete_docs:
                status_msg += "\n🎉 Knowledge base completely cleared, including all original documents!"
            else:
                status_msg += "\n🎉 Knowledge base completely cleared, original documents preserved!"
            
            status_msg += "\n📊 Cleanup scope:"
            status_msg += "\n  • ✅ ChromaDB vector data"
            status_msg += "\n  • ✅ Index files"
            status_msg += "\n  • ✅ MinerU processing cache"
            if delete_docs:
                status_msg += "\n  • ✅ Original uploaded documents"
            else:
                status_msg += "\n  • ⬜ Original uploaded documents (preserved)"
            
            return status_msg, get_collection_info()
            
        except Exception as e:
            error_msg = f"❌ Serious error during complete knowledge base cleanup:\n{str(e)}\n"
            error_msg += "📋 Suggest manually checking these directories:\n"
            error_msg += f"  • {CHROMA_PERSIST_DIR}\n"
            error_msg += f"  • {UPLOAD_DIR}\n"
            return error_msg, get_collection_info()
    
    # Stream Q&A processing
    def stream_answer(query, top_k):
        """Stream generate answer"""
        if not query:
            yield "Please enter a question", ""
            return
        
        for answer, sources in query_knowledge_base_stream(query, top_k):
            sources_text = "Information sources:\n" + "\n".join([f"• {os.path.basename(s)}" for s in sources]) if sources else "No reference sources"
            yield answer, sources_text
    
    # Event bindings
    upload_btn.click(
        fn=upload_file_handler,
        inputs=[file_input, use_mineru_checkbox],
        outputs=[upload_status, collection_info]
    )
    
    load_from_disk_btn.click(
        fn=load_from_disk_handler,
        inputs=[dir_input, use_mineru_checkbox],
        outputs=[upload_status, collection_info]
    )
    
    refresh_btn.click(fn=get_collection_info, outputs=collection_info)
    clear_btn.click(fn=clear_knowledge_base, outputs=[upload_status, collection_info])
    confirm_clear_btn.click(
        fn=clear_knowledge_base_completely, 
        inputs=[delete_docs_checkbox], 
        outputs=[upload_status, collection_info]
    )
    
    # Stream Q&A events
    query_btn.click(
        fn=stream_answer,
        inputs=[query_input, top_k_slider],
        outputs=[answer_output, sources_output]
    )
    
    # Initialize
    demo.load(fn=get_collection_info, outputs=collection_info)

if __name__ == "__main__":
    print(f"\n{'='*60}")
    print("🚀 Streaming RAG System Started Successfully!")
    print(f"{'='*60}")
    print(f"GPU Acceleration: {'✅ Enabled' if torch.cuda.is_available() else '❌ Disabled'}")
    print(f"Device: {device}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"LLM Model: {LLM_MODEL}")
    print(f"Streaming Output: ✅ Enabled")
    print(f"{'='*60}\n")
    
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False) 