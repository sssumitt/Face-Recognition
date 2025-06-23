# Face Recognition Model with ArcFace Loss

A PyTorch-based face-recognition framework that uses ArcFace (Additive Angular Margin) loss to produce highly discriminative embeddings.

## Features

- **ArcFace Loss**: Implements the additive angular margin penalty for improved class separability during training.  
- **Embedding Extraction**: Given an aligned face image, outputs a fixed-length embedding vector.  
- **Similarity Scoring**: Computes cosine similarity between embeddings for verification or identification.  
- **Adaptive Updates**: Optionally augment user profiles with new embeddings on successful verification.  

