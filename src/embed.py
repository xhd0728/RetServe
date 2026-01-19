from infinity_emb import EngineArgs, AsyncEngineArray
from tqdm import tqdm
import jsonlines
import numpy as np
import os
import gc
import asyncio
import argparse
from typing import List, Dict, Any, Optional
from src.config_loader import config_loader
from src.logging import logger


class EmbeddingProcessor:
    """Embedding processor for text corpora"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize embedding processor
        
        Args:
            config: Configuration dictionary
        """
        self.model_config = config["model"]
        self.data_config = config["data"]
        
        # Set CUDA visible devices
        os.environ["CUDA_VISIBLE_DEVICES"] = self.model_config["gpu_ids"]
    
    async def load_corpus(self) -> List[str]:
        """
        Load corpus from JSONL file
        
        Returns:
            List of text contents
        """
        logger.info(f"Loading corpus from {self.data_config['corpus_path']}")
        contents = []
        
        with jsonlines.open(self.data_config["corpus_path"], mode="r") as reader:
            for item in reader:
                contents.append(item["contents"])
        
        logger.info(f"Loaded {len(contents)} documents from corpus")
        return contents
    
    async def create_embeddings(self, data: List[str]) -> np.ndarray:
        """
        Create embeddings using infinity_emb
        
        Args:
            data: List of text strings
            
        Returns:
            Embeddings as numpy array
        """
        logger.info(f"Starting embedding with batch size {self.model_config['batch_size']}")
        
        # Create engine arguments
        engine_args = EngineArgs(
            model_name_or_path=self.model_config["path"],
            batch_size=self.model_config["batch_size"],
            bettertransformer=self.model_config["bettertransformer"],
            pooling_method=self.model_config["pooling_method"],
            device=self.model_config["device"],
            model_warmup=self.model_config["model_warmup"],
            trust_remote_code=self.model_config["trust_remote_code"],
        )
        
        # Initialize model
        model = AsyncEngineArray.from_args([engine_args])[0]
        
        embeddings = []
        eff_bs = self.model_config["batch_size"] * len(self.model_config["gpu_ids"].split(","))
        n = len(data)
        
        async with model:
            with tqdm(total=n, desc="[infinity] Embedding: ") as pbar:
                for i in range(0, n, eff_bs):
                    chunk = data[i : i + eff_bs]
                    vecs, _ = await model.embed(sentences=chunk)
                    embeddings.extend(vecs)
                    pbar.update(len(chunk))
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        logger.info(f"Created embeddings of shape {embeddings_array.shape}")
        return embeddings_array
    
    def save_embeddings(self, embeddings: np.ndarray) -> None:
        """
        Save embeddings to file
        
        Args:
            embeddings: Embeddings to save
        """
        os.makedirs(os.path.dirname(self.data_config["embedding_path"]), exist_ok=True)
        np.save(self.data_config["embedding_path"], embeddings)
        logger.info(f"Saved embedding to {self.data_config['embedding_path']}")
    
    async def process(self) -> None:
        """
        Main processing method
        """
        try:
            # Load corpus
            data = await self.load_corpus()
            
            # Create embeddings
            embeddings = await self.create_embeddings(data)
            
            # Save embeddings
            self.save_embeddings(embeddings)
            
            # Clean up resources
            del embeddings
            gc.collect()
            logger.info("Embedding process completed successfully")
            
        except Exception as e:
            logger.error(f"Error during embedding process: {str(e)}")
            raise


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Corpus Embedding Tool")
    parser.add_argument(
        "--config", 
        type=str, 
        default="embed", 
        help="Configuration file name (without extension)"
    )
    return parser.parse_args()


async def main():
    """
    Main function
    """
    args = parse_args()
    
    # Load configuration
    config = config_loader.load_config(args.config)
    
    # Create and run embedding processor
    processor = EmbeddingProcessor(config)
    await processor.process()


if __name__ == "__main__":
    asyncio.run(main())
