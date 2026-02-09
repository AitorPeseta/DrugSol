#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Make Embeddings ChemBERTa: Molecular Embedding Generation
==========================================================

Generates molecular embeddings using ChemBERTa, a RoBERTa-based transformer
model pre-trained on millions of molecules. These embeddings capture learned
chemical knowledge that complements traditional molecular descriptors.

Strategy: Frozen Embeddings (Feature Extraction)
    Instead of fine-tuning ChemBERTa for solubility prediction, we use it
    as a "chemical knowledge extractor". The pre-trained model has learned
    fundamental chemistry from 77M molecules - we leverage this knowledge
    by extracting embeddings and feeding them to gradient boosting models.

Why This Works:
    - ChemBERTa "knows" chemistry: atom relationships, functional groups, etc.

Arguments:
    --input          : Input Parquet/CSV file with molecular data
    --output         : Output Parquet file path
    --smiles-col     : Column name containing SMILES strings (default: smiles_neutral)
    --model-name     : HuggingFace model ID (default: seyonec/ChemBERTa-zinc-base-v1)
    --batch-size     : Batch size for inference (default: 32)
    --embed-prefix   : Prefix for embedding column names (default: bert_)
    --pooling        : Pooling strategy: cls, mean, max (default: cls)
    --device         : Device to use: auto, cpu, cuda (default: auto)

Usage:
    python make_embeddings_chemberta.py \\
        --input train_data.parquet \\
        --output train_chemberta_embeddings.parquet \\
        --smiles-col smiles_neutral \\
        --model-name "seyonec/ChemBERTa-zinc-base-v1" \\
        --batch-size 32 \\
        --embed-prefix "bert_" \\
        --pooling cls \\
        --device auto

Output:
    Parquet file with original columns plus embedding columns:
    - bert_0, bert_1, ..., bert_767 (768 dimensions for base models)
    
    For invalid SMILES, embeddings are set to zeros.

Model Options:
    - seyonec/ChemBERTa-zinc-base-v1   : Trained on ZINC dataset (good default)
    - seyonec/ChemBERTa-zinc250k-v1    : Trained on ZINC250k subset
    - DeepChem/ChemBERTa-77M-MLM       : Largest model (77M molecules from PubChem)
    - DeepChem/ChemBERTa-77M-MTR       : Multi-task regression variant

Notes:
    - Batch size should be adjusted based on available memory:
      * CPU: 16-32
      * GPU (8GB): 64-128
      * GPU (16GB+): 256-512
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings from transformers
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Lazy imports for transformers (heavy library)
torch = None
AutoTokenizer = None
AutoModel = None


def lazy_import_transformers():
    """
    Lazily import PyTorch and Transformers to speed up script loading
    and provide better error messages if dependencies are missing.
    """
    global torch, AutoTokenizer, AutoModel
    
    try:
        import torch as _torch
        torch = _torch
    except ImportError:
        sys.exit(
            "[ERROR] PyTorch not found. Install with:\n"
            "  pip install torch\n"
            "  or: conda install pytorch -c pytorch"
        )
    
    try:
        from transformers import AutoTokenizer as _AutoTokenizer
        from transformers import AutoModel as _AutoModel
        AutoTokenizer = _AutoTokenizer
        AutoModel = _AutoModel
    except ImportError:
        sys.exit(
            "[ERROR] Transformers library not found. Install with:\n"
            "  pip install transformers\n"
            "  or: conda install transformers -c huggingface"
        )


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

class ChemBERTaEmbedder:
    """
    Wrapper class for ChemBERTa embedding extraction.
    
    Handles model loading, tokenization, and batch inference with proper
    memory management and progress tracking.
    """
    
    def __init__(
        self,
        model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
        device: str = "auto",
        pooling: str = "cls"
    ):
        """
        Initialize the ChemBERTa embedder.
        
        Args:
            model_name: HuggingFace model ID
            device: "auto", "cpu", or "cuda"
            pooling: "cls" (CLS token), "mean" (mean pooling), or "max" (max pooling)
        """
        lazy_import_transformers()
        
        self.model_name = model_name
        self.pooling = pooling
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        print(f"[ChemBERTa] Loading model: {model_name}")
        print(f"[ChemBERTa] Device: {self.device}")
        print(f"[ChemBERTa] Pooling: {pooling}")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Move model to device and set to evaluation mode
        self.model.to(self.device)
        self.model.eval()
        
        # Get embedding dimension
        self.embed_dim = self.model.config.hidden_size
        print(f"[ChemBERTa] Embedding dimension: {self.embed_dim}")
    
    def _pool_embeddings(self, last_hidden_state, attention_mask):
        """
        Apply pooling strategy to get fixed-size embedding.
        
        Args:
            last_hidden_state: Tensor of shape (batch, seq_len, hidden_size)
            attention_mask: Tensor of shape (batch, seq_len)
        
        Returns:
            Tensor of shape (batch, hidden_size)
        """
        if self.pooling == "cls":
            # Use [CLS] token (first token)
            return last_hidden_state[:, 0, :]
        
        elif self.pooling == "mean":
            # Mean pooling over non-padded tokens
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
            sum_embeddings = torch.sum(last_hidden_state * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            return sum_embeddings / sum_mask
        
        elif self.pooling == "max":
            # Max pooling over sequence dimension
            mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size())
            last_hidden_state[mask == 0] = -1e9  # Mask padding with large negative
            return torch.max(last_hidden_state, dim=1)[0]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def embed_batch(self, smiles_list: list) -> np.ndarray:
        """
        Generate embeddings for a batch of SMILES strings.
        
        Args:
            smiles_list: List of SMILES strings
        
        Returns:
            NumPy array of shape (batch_size, embed_dim)
        """
        # Tokenize
        inputs = self.tokenizer(
            smiles_list,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Forward pass (no gradient computation)
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Pool embeddings
        embeddings = self._pool_embeddings(
            outputs.last_hidden_state,
            inputs["attention_mask"]
        )
        
        # Move to CPU and convert to numpy
        return embeddings.cpu().numpy()
    
    def embed_dataframe(
        self,
        df: pd.DataFrame,
        smiles_col: str,
        batch_size: int = 32,
        embed_prefix: str = "bert_"
    ) -> pd.DataFrame:
        """
        Generate embeddings for all molecules in a DataFrame.
        
        Args:
            df: Input DataFrame
            smiles_col: Column name containing SMILES
            batch_size: Number of molecules per batch
            embed_prefix: Prefix for embedding column names
        
        Returns:
            DataFrame with original columns plus embedding columns
        """
        n_samples = len(df)
        smiles_series = df[smiles_col].fillna("").astype(str)
        
        # Pre-allocate embedding matrix
        all_embeddings = np.zeros((n_samples, self.embed_dim), dtype=np.float32)
        
        # Track valid/invalid SMILES
        valid_count = 0
        invalid_indices = []
        
        # Process in batches
        print(f"[ChemBERTa] Processing {n_samples:,} molecules...")
        
        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Embedding"):
            end_idx = min(start_idx + batch_size, n_samples)
            batch_smiles = smiles_series.iloc[start_idx:end_idx].tolist()
            
            # Filter out empty/invalid SMILES for this batch
            valid_mask = [bool(s.strip()) for s in batch_smiles]
            valid_smiles = [s for s, v in zip(batch_smiles, valid_mask) if v]
            
            if valid_smiles:
                try:
                    # Get embeddings for valid SMILES
                    batch_embeddings = self.embed_batch(valid_smiles)
                    
                    # Map back to original indices
                    valid_idx = 0
                    for i, is_valid in enumerate(valid_mask):
                        global_idx = start_idx + i
                        if is_valid:
                            all_embeddings[global_idx] = batch_embeddings[valid_idx]
                            valid_idx += 1
                            valid_count += 1
                        else:
                            invalid_indices.append(global_idx)
                
                except Exception as e:
                    # If batch fails, mark all as invalid
                    print(f"[WARN] Batch {start_idx}-{end_idx} failed: {e}")
                    for i in range(start_idx, end_idx):
                        invalid_indices.append(i)
            else:
                # All SMILES in batch are invalid
                for i in range(start_idx, end_idx):
                    invalid_indices.append(i)
        
        # Report statistics
        n_invalid = len(invalid_indices)
        if n_invalid > 0:
            print(f"[WARN] {n_invalid:,} molecules failed ({n_invalid/n_samples*100:.1f}%)")
        print(f"[ChemBERTa] Successfully embedded {valid_count:,} molecules")
        
        # Create embedding DataFrame
        embed_cols = [f"{embed_prefix}{i}" for i in range(self.embed_dim)]
        df_embeddings = pd.DataFrame(
            all_embeddings,
            columns=embed_cols,
            index=df.index
        )
        
        # Concatenate with original DataFrame
        result = pd.concat([df, df_embeddings], axis=1)
        
        return result


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main entry point for ChemBERTa embedding generation."""
    
    ap = argparse.ArgumentParser(
        description="Generate ChemBERTa molecular embeddings."
    )
    
    # Input/Output
    ap.add_argument("--input", "-i", required=True,
                    help="Input Parquet/CSV file")
    ap.add_argument("--output", "-o", required=True,
                    help="Output Parquet file")
    ap.add_argument("--smiles-col", default="smiles_neutral",
                    help="SMILES column name (default: smiles_neutral)")
    
    # Model configuration
    ap.add_argument("--model-name", default="seyonec/ChemBERTa-zinc-base-v1",
                    help="HuggingFace model ID (default: seyonec/ChemBERTa-zinc-base-v1)")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Batch size for inference (default: 32)")
    ap.add_argument("--embed-prefix", default="bert_",
                    help="Prefix for embedding columns (default: bert_)")
    ap.add_argument("--pooling", choices=["cls", "mean", "max"], default="cls",
                    help="Pooling strategy (default: cls)")
    ap.add_argument("--device", default="auto",
                    help="Device: auto, cpu, cuda (default: auto)")
    
    args = ap.parse_args()
    
    # -------------------------------------------------------------------------
    # Load Input Data
    # -------------------------------------------------------------------------
    print(f"[ChemBERTa] Loading {args.input}...")
    
    if args.input.endswith(".parquet"):
        df = pd.read_parquet(args.input)
    else:
        df = pd.read_csv(args.input)
    
    if args.smiles_col not in df.columns:
        sys.exit(f"[ERROR] SMILES column '{args.smiles_col}' not found in input file.")
    
    print(f"[ChemBERTa] Loaded {len(df):,} rows")
    
    # -------------------------------------------------------------------------
    # Initialize Embedder
    # -------------------------------------------------------------------------
    embedder = ChemBERTaEmbedder(
        model_name=args.model_name,
        device=args.device,
        pooling=args.pooling
    )
    
    # -------------------------------------------------------------------------
    # Generate Embeddings
    # -------------------------------------------------------------------------
    df_result = embedder.embed_dataframe(
        df=df,
        smiles_col=args.smiles_col,
        batch_size=args.batch_size,
        embed_prefix=args.embed_prefix
    )
    
    # -------------------------------------------------------------------------
    # Save Output
    # -------------------------------------------------------------------------
    print(f"[ChemBERTa] Saving to {args.output}...")
    df_result.to_parquet(args.output, index=False)
    
    # Print summary
    n_embed_cols = embedder.embed_dim
    print(f"\n[ChemBERTa] Summary:")
    print(f"   -> Input rows: {len(df):,}")
    print(f"   -> Output rows: {len(df_result):,}")
    print(f"   -> Embedding dimension: {n_embed_cols}")
    print(f"   -> New columns: {args.embed_prefix}0 to {args.embed_prefix}{n_embed_cols-1}")
    print(f"   -> Total columns: {len(df_result.columns):,}")


if __name__ == "__main__":
    main()
