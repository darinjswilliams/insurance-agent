from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
import PyPDF2
from pathlib import Path
import os

from app.utils.config import config
from app.utils.logger import logger
from openai import OpenAI
class PolicyVectorStore:
    """Manages Pinecone collection for insurance policy documents"""
    
    def __init__(self):
        logger.info("Initializing Pinecone vector store")
        
        # Initialize ChromaDB client
        #Initialize Pinecone client 
        self.pc = Pinecone(api_key=config.pinecone_api_key)

        # Ensure index exists 
        self.index_name = config.pinecone_index_name

        if self.index_name not in [idx["name"] for idx in self.pc.list_indexes()]:
         logger.info(f"Creating Pinecone index: {self.index_name}") 
         self.pc.create_index( 
            name=self.index_name,
            dimension=1536, # match your embedding model 
            metric="cosine",
            spec=ServerlessSpec( cloud="aws", region="us-east-1" 
            )
        )

        self.index = self.pc.Index(self.index_name)

        logger.info(f"Using OpenAI embedding model: {config.embedding_model}")
        
        # OpenAI client for embeddings
        self.openai = OpenAI(api_key=config.openai_api_key)
        self.embedding_model = config.embedding_model
        
        logger.info(f"Using OpenAI embedding model: {self.embedding_model}")
        logger.info(f"Pinecone index '{self.index_name}' ready")
    
    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts using OpenAI"""
        
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=texts
        )
        return [item.embedding for item in response.data]


    def _embed_query(self, query: str) -> List[float]:
        """Embed a single query string"""
        response = self.openai.embeddings.create(
            model=self.embedding_model,
            input=query
        )
        return response.data[0].embedding



    def load_pdf_policy(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text chunks from policy PDF"""
        logger.info(f"Loading policy PDF: {pdf_path}")
        
        if not Path(pdf_path).exists():
            logger.error(f"Policy PDF not found: {pdf_path}")
            return []
        
        chunks = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                logger.info(f"PDF has {len(pdf_reader.pages)} pages")
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    text = page.extract_text()
                    # Split into smaller chunks (500 chars with 50 char overlap)
                    chunk_size = 500
                    overlap = 50
                    
                    for i in range(0, len(text), chunk_size - overlap):
                        chunk = text[i:i + chunk_size]
                        if len(chunk.strip()) > 50:
                            chunks.append({
                                "text": chunk.strip(),
                                "page": page_num,
                                "source": pdf_path
                            })
            
            logger.info(f"Extracted {len(chunks)} chunks from PDF")
            return chunks
        
        except Exception as e:
            logger.error(f"Error loading PDF: {e}")
            return []
    
    def populate_from_pdf(self, pdf_path: Optional[str] = None):
        """Load policy PDF into vector store"""
        pdf_path = pdf_path or config.policy_pdf_path
        
        # Check if already populated
        stats = self.index.describe_index_stats()

        if stats.get("total_vector_count", 0) > 0:
            logger.info("Index already populated. Skipping.")
            return

        # Load PDF into chunks
        chunks = self.load_pdf_policy(pdf_path)
        if not chunks:
            logger.warning("No chunks extracted from PDF")
            return
        
        logger.info("Adding chunks to vector store (embeddings generated automatically)")

        # 1. Extract raw text
        texts = [chunk["text"] for chunk in chunks]

        # 2. Generate embeddings (Pinecone does NOT do this automatically)
        embeddings = self._embed_texts(texts)

        # 3. Build Pinecone vector objects
        vectors =[]
        for i, text in enumerate(texts):
            vectors.append({
                "id": f"chunk_{i}",
                "values": embeddings[i],
                "metadata": {
                    "text": chunks[i]["text"],
                    "page": chunks[i]["page"],
                    "source": chunks[i]["source"]
                }
            })

        #Upload to Pinecone
        self.index.upsert(vectors)
    

        logger.info(f"Successfully added {len(chunks)} chunks to Pinecone")
    
    def retrieve(self, query: str, top_k: int = 5) -> str:
        """Retrieve relevant policy text for a query"""
        logger.info(f"Retrieving policy text for query: {query[:100]}...")
        
        # ✅ 1. Embed the query manually (Pinecone does NOT auto-embed)
        query_embedding = self._embed_query(query)

        # 2. Query Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )


        # 3. Handle no results
        if not results.matches:
            logger.warning("No relevant policy documents found")
            return
      
        # 4. Extract the text from metadata
        retrieved_chunks = [match.metadata["text"] for match in results.matches]
       
        logger.info(f"Retrieved {len(retrieved_chunks)} relevant chunks")
        
        # 5. Return combined text (same behavior as Chroma)
        return "\n\n".join(retrieved_chunks)

    def count(self) -> int: 
        """Return total number of vectors stored in the Pinecone index.""" 
        stats = self.index.describe_index_stats()
        return stats.get("total_vector_count", 0)

# Global vector store instance
policy_store = PolicyVectorStore()
