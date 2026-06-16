# Transformer Summarizer

## Course Information
- **Course:** Course 4 - Natural Language Processing (Attention Models)
- **Platform:** DeepLearning.AI
- **Assignment:** #2 - Transformer for Text Summarization

## 📌 Overview
This assignment focused on building a **Transformer-based summarization model** using an encoder-decoder architecture. Unlike the lecture examples, I implemented the **full encoder-decoder transformer** from scratch (with the encoder provided, decoder implemented by me) to summarize dialogues and articles.

The key insight is that attention mechanisms allow the model to focus on relevant parts of the input when generating each word of the summary, enabling parallel processing and superior performance compared to RNN-based sequence-to-sequence models.

## 🎯 What I Learned
- **Scaled Dot-Product Attention:** Implemented the core attention mechanism that powers the entire transformer: `Attention(Q,K,V) = softmax(QK^T/√dk + M)V`
- **Positional Encoding:** Understood why we need to add positional information to embeddings (since transformers process all tokens in parallel) and implemented sinusoidal positional encodings.
- **Masking Mechanisms:** 
  - **Padding masks:** Prevent the model from attending to padding tokens
  - **Look-ahead masks:** Prevent the decoder from "seeing" future tokens during training (causal attention)
- **Encoder Architecture:** Understood how the encoder stacks multi-head self-attention with feed-forward networks, using residual connections and layer normalization.
- **Decoder Architecture:** Implemented the decoder with:
  - Masked self-attention (causal)
  - Cross-attention (encoder-decoder attention)
  - Feed-forward networks with residual connections
- **Transformer Training:** Used a custom learning rate schedule (warmup + decay) and trained the model with greedy decoding for inference.
- **Greedy Decoding:** Implemented word-by-word prediction for generating summaries at inference time.

## ⚙️ Key Skills Practiced
- Implementing self-attention and multi-head attention
- Building transformer encoder and decoder layers
- Creating custom TensorFlow layers with `call()` methods
- Handling masks (padding and look-ahead)
- Training with custom training loops
- Text preprocessing and tokenization for transformers
- Learning rate scheduling (warmup + rsqrt decay)
- Inference with greedy decoding

## 🚧 Challenges Faced
1. **Understanding Attention Dimensions:** Keeping track of Q, K, V shapes through the attention mechanism (batch_size, seq_len, d_model) and ensuring matrix multiplications aligned correctly.
2. **Implementing the Decoder Layer:** 
   - Block 1: Masked self-attention with look-ahead mask
   - Block 2: Cross-attention with encoder output (Q from block 1, K/V from encoder)
   - Block 3: Feed-forward with residual connections
3. **Mask Broadcasting:** Understanding how masks are broadcasted across batch and head dimensions in multi-head attention.
4. **Residual Connections & Layer Normalization:** Getting the order right: attention → add → normalize, repeated for each sub-layer.
5. **Inference Logic:** Implementing `next_word()` with the correct masks and ensuring the model is in inference mode (`training=False`).
6. **Overfitting on Small Dataset:** The model memorized training examples due to small dataset size (20 epochs), performing well on training but poorly on test data.
7. **Custom Learning Rate Schedule:** Understanding the `CustomSchedule` class with warmup steps for faster convergence.

## 📋 Important Submission Notes (AutoGrader)
To avoid `Grader Error: Grader feedback not found` or similar unexpected errors from the DeepLearning.AI AutoGrader, I made sure NOT to:
- Add any extra `print` statements in the assignment
- Add any extra code cells
- Change any of the function parameters
- Use global variables inside graded exercises (unless specifically instructed)
- Change the assignment code where not required (like creating extra variables)

> 💡 **Tip:** If you run into unexpected grader errors, check the above points first. You can always get a fresh copy of the assignment by following the workspace refresh instructions on DeepLearning.AI.

## 🛠️ Technologies & Tools
- TensorFlow 2.x / Keras
- NumPy
- Pandas
- Matplotlib
- TensorFlow's `MultiHeadAttention` layer
- Custom layers with `tf.keras.layers.Layer`
- Text preprocessing (Tokenizer, padding)

## 📈 Results
- Successfully built a working transformer encoder-decoder for text summarization.
- Achieved near-perfect reconstruction of training examples (indicating overfitting) after 20 epochs.
- Demonstrated the model's ability to summarize dialogues into concise summaries.

### Sample Output
**Training Set Example:**

Dialogue: amanda: I baked cookies. will: Okay. amanda: Do you want some? will: Sure, thank you.
Human summary: amanda bakes cookies and offers will some.
Model summary: amanda bakes cookies and offers will some.