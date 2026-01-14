#!/usr/bin/env python3
"""
ONNX Inference Module for ISC21 Image Embedding.

Minimal dependencies: onnxruntime, PIL, numpy

Usage:
    from onnx_inference import ISCEmbedder

    embedder = ISCEmbedder("isc_model.onnx")
    embedding = embedder.embed("image.jpg")  # Returns (256,) numpy array
    embeddings = embedder.embed_batch(["img1.jpg", "img2.jpg"])  # Returns (N, 256)
"""

from pathlib import Path
from typing import Union, List, Optional
import json

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    raise ImportError("onnxruntime is required. Install with: pip install onnxruntime")


class ISCEmbedder:
    """
    ONNX-based image embedding extractor for ISC21 model.

    Produces 256-dimensional L2-normalized embeddings for image similarity.

    Args:
        model_path: Path to ONNX model file
        config_path: Path to config JSON (default: same name as model with .json)
        providers: ONNX Runtime execution providers (default: auto-detect)

    Example:
        >>> embedder = ISCEmbedder("isc_model.onnx")
        >>> vec = embedder.embed("photo.jpg")
        >>> print(vec.shape)  # (256,)
        >>> print(np.linalg.norm(vec))  # ~1.0 (L2 normalized)
    """

    # Default config (for isc_ft_v107)
    DEFAULT_CONFIG = {
        "input_size": 512,
        "mean": [0.5, 0.5, 0.5],
        "std": [0.5, 0.5, 0.5],
        "fc_dim": 256,
    }

    def __init__(
        self,
        model_path: str,
        config_path: Optional[str] = None,
        providers: Optional[List[str]] = None,
    ):
        self.model_path = Path(model_path)

        # Load config
        if config_path is None:
            config_path = self.model_path.with_suffix(".json")

        if Path(config_path).exists():
            with open(config_path) as f:
                self.config = json.load(f)
        else:
            print(f"Config not found at {config_path}, using defaults")
            self.config = self.DEFAULT_CONFIG.copy()

        self.input_size = self.config["input_size"]
        self.mean = np.array(self.config["mean"], dtype=np.float32)
        self.std = np.array(self.config["std"], dtype=np.float32)
        self.fc_dim = self.config.get("fc_dim", 256)

        # Setup ONNX Runtime session
        if providers is None:
            providers = self._get_default_providers()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            str(self.model_path),
            sess_options=sess_options,
            providers=providers,
        )

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _get_default_providers(self) -> List[str]:
        """Get available execution providers, preferring GPU."""
        available = ort.get_available_providers()
        preferred = [
            "CUDAExecutionProvider",
            "CoreMLExecutionProvider",
            "CPUExecutionProvider",
        ]
        return [p for p in preferred if p in available]

    def preprocess(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Preprocess a single image for inference.

        Args:
            image: File path or PIL Image

        Returns:
            Preprocessed image as (1, 3, H, W) float32 array
        """
        if isinstance(image, (str, Path)):
            img = Image.open(image).convert("RGB")
        else:
            img = image.convert("RGB")

        # Resize to input_size x input_size
        img = img.resize((self.input_size, self.input_size), Image.BILINEAR)

        # Convert to numpy array and normalize
        arr = np.array(img, dtype=np.float32) / 255.0  # [0, 1]

        # Normalize with mean and std (per-channel)
        arr = (arr - self.mean) / self.std

        # HWC -> CHW and add batch dimension
        arr = arr.transpose(2, 0, 1)  # (3, H, W)
        arr = arr[np.newaxis, ...]  # (1, 3, H, W)

        return arr.astype(np.float32)

    def preprocess_batch(
        self, images: List[Union[str, Path, Image.Image]]
    ) -> np.ndarray:
        """
        Preprocess multiple images for batch inference.

        Args:
            images: List of file paths or PIL Images

        Returns:
            Preprocessed images as (N, 3, H, W) float32 array
        """
        batch = [self.preprocess(img)[0] for img in images]
        return np.stack(batch, axis=0)

    def embed(self, image: Union[str, Path, Image.Image]) -> np.ndarray:
        """
        Extract embedding vector from a single image.

        Args:
            image: File path or PIL Image

        Returns:
            Embedding vector as (fc_dim,) numpy array (L2 normalized)
        """
        input_tensor = self.preprocess(image)
        outputs = self.session.run([self.output_name], {self.input_name: input_tensor})
        return outputs[0][0]  # Remove batch dimension

    def embed_batch(
        self,
        images: List[Union[str, Path, Image.Image]],
        batch_size: int = 32,
    ) -> np.ndarray:
        """
        Extract embeddings from multiple images.

        Args:
            images: List of file paths or PIL Images
            batch_size: Maximum batch size for inference

        Returns:
            Embeddings as (N, fc_dim) numpy array
        """
        all_embeddings = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            input_tensor = self.preprocess_batch(batch_images)
            outputs = self.session.run(
                [self.output_name], {self.input_name: input_tensor}
            )
            all_embeddings.append(outputs[0])

        return np.concatenate(all_embeddings, axis=0)

    def __call__(
        self, image: Union[str, Path, Image.Image, List]
    ) -> np.ndarray:
        """
        Extract embeddings (convenience method).

        Args:
            image: Single image or list of images

        Returns:
            Embedding(s) as numpy array
        """
        if isinstance(image, list):
            return self.embed_batch(image)
        return self.embed(image)

    @property
    def embedding_dim(self) -> int:
        """Return embedding dimension."""
        return self.fc_dim


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Since embeddings are L2-normalized, this equals the dot product.
    """
    return float(np.dot(a, b))


def cosine_similarity_matrix(
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray
) -> np.ndarray:
    """
    Compute pairwise cosine similarities between two sets of embeddings.

    Args:
        embeddings_a: (N, D) array
        embeddings_b: (M, D) array

    Returns:
        (N, M) similarity matrix
    """
    return embeddings_a @ embeddings_b.T


def find_similar(
    query_embedding: np.ndarray,
    reference_embeddings: np.ndarray,
    top_k: int = 10,
) -> tuple:
    """
    Find most similar embeddings to a query.

    Args:
        query_embedding: (D,) query vector
        reference_embeddings: (N, D) reference vectors
        top_k: Number of results to return

    Returns:
        (indices, similarities) tuple
    """
    similarities = reference_embeddings @ query_embedding
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return top_indices, similarities[top_indices]


# CLI interface
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract image embeddings using ONNX model"
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to ONNX model file",
    )
    parser.add_argument(
        "images",
        type=str,
        nargs="+",
        help="Image file(s) to process",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for embeddings (.npy)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for inference (default: 32)",
    )
    args = parser.parse_args()

    # Create embedder
    embedder = ISCEmbedder(args.model_path)
    print(f"Model loaded: {args.model_path}")
    print(f"Input size: {embedder.input_size}x{embedder.input_size}")
    print(f"Embedding dim: {embedder.embedding_dim}")

    # Process images
    if len(args.images) == 1:
        embedding = embedder.embed(args.images[0])
        print(f"\nEmbedding for {args.images[0]}:")
        print(f"  Shape: {embedding.shape}")
        print(f"  Norm: {np.linalg.norm(embedding):.6f}")
        print(f"  First 5 values: {embedding[:5]}")
        embeddings = embedding[np.newaxis, :]
    else:
        embeddings = embedder.embed_batch(args.images, batch_size=args.batch_size)
        print(f"\nProcessed {len(args.images)} images")
        print(f"  Embeddings shape: {embeddings.shape}")

        # Show pairwise similarities
        if len(args.images) <= 10:
            print("\nPairwise similarities:")
            sim_matrix = cosine_similarity_matrix(embeddings, embeddings)
            for i in range(len(args.images)):
                for j in range(i + 1, len(args.images)):
                    print(f"  {Path(args.images[i]).name} <-> {Path(args.images[j]).name}: {sim_matrix[i, j]:.4f}")

    # Save embeddings
    if args.output:
        np.save(args.output, embeddings)
        print(f"\nEmbeddings saved to: {args.output}")
