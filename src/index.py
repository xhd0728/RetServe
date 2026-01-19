import os
import faiss
import numpy as np
from tqdm import tqdm
import argparse
from typing import Dict, Any, Optional
from src.config_loader import config_loader
from src.logging import logger


class IndexBuilder:
    """FAISS index builder for embeddings"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize index builder
        
        Args:
            config: Configuration dictionary
        """
        self.index_config = config["index"]
        self.data_config = config["data"]
    
    def load_embeddings(self) -> np.ndarray:
        """
        Load embeddings from numpy file
        
        Returns:
            Embeddings as numpy array
        """
        logger.info(f"Loading embeddings from {self.data_config['embedding_path']}")
        embedding = np.load(self.data_config["embedding_path"])
        logger.info(f"Loaded embeddings of shape {embedding.shape}")
        return embedding
    
    def build_index(self, embedding: np.ndarray) -> faiss.Index:
        """
        Build FAISS index from embeddings
        
        Args:
            embedding: Embeddings to index
            
        Returns:
            Built FAISS index (always CPU-based for efficient saving)
        """
        dim = embedding.shape[1]
        vec_ids = np.arange(embedding.shape[0]).astype(np.int64)
        
        # Create CPU index
        logger.info(f"Building FAISS index with dim={dim}")
        cpu_flat = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIDMap2(cpu_flat)
        
        total = embedding.shape[0]
        logger.info(f"Indexing {total} vectors")
        
        with tqdm(
            total=total,
            desc="[faiss] Indexing: ",
            unit="vec",
        ) as pbar:
            for start in range(0, total, self.index_config["chunk_size"]):
                end = min(start + self.index_config["chunk_size"], total)
                cpu_index.add_with_ids(embedding[start:end], vec_ids[start:end])
                pbar.update(end - start)
        
        # Always return CPU index for efficient saving
        return cpu_index
    
    def save_index(self, index: faiss.Index) -> None:
        """
        Save FAISS index to file
        
        Args:
            index: FAISS index to save
        """
        os.makedirs(os.path.dirname(self.data_config["index_path"]), exist_ok=True)
        
        # Always try to convert to CPU index before saving
        # This handles both GPU and CPU indices safely
        try:
            # First check if it's already a CPU index
            if hasattr(index, 'index') and hasattr(index.index, '__class__'):
                # This is likely a GpuIndex that wraps a CPU index
                index = faiss.index_gpu_to_cpu(index)
            elif hasattr(index, '__class__'):
                # Check class name directly
                class_name = index.__class__.__name__
                if 'Gpu' in class_name:
                    index = faiss.index_gpu_to_cpu(index)
        except Exception as e:
            logger.warning(f"Could not determine if index is GPU type: {e}")
            logger.info("Treating as CPU index")
        
        faiss.write_index(index, self.data_config["index_path"])
        logger.info(f"Saved index to {self.data_config['index_path']}")
    
    def build(self) -> None:
        """
        Main index building method
        """
        try:
            # Load embeddings
            embedding = self.load_embeddings()
            
            # Build index
            faiss_index = self.build_index(embedding)
            
            # Save index
            self.save_index(faiss_index)
            
            logger.info("Indexing process completed successfully")
            
        except Exception as e:
            logger.error(f"Error during indexing process: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Corpus Indexing Tool")
    parser.add_argument(
        "--config", 
        type=str, 
        default="index", 
        help="Configuration file name (without extension)"
    )
    return parser.parse_args()


def main():
    """
    Main function
    """
    args = parse_args()
    
    # Load configuration
    config = config_loader.load_config(args.config)
    
    # Create and run index builder
    builder = IndexBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()
