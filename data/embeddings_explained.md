# Embeddings Explained

An embedding is a list of numbers, a vector, that represents the meaning of a piece of text. A sentence about
"refund policy" and a sentence about "getting my money back" use different words, yet their embeddings point in
nearly the same direction, because their meaning is close.

## Local embedding models

This project uses the SentenceTransformers model all-MiniLM-L6-v2. It runs on your own computer, costs nothing and
turns every chunk into a vector with 384 numbers. The model is downloaded once, about ninety megabytes, and then
works offline. No text is sent to an external service to create embeddings.

## Cosine similarity

To compare two embeddings we measure the angle between them. Cosine similarity is 1 when the vectors point the same way,
around 0 when the texts are unrelated, and it can even be negative. The vector database reports a cosine distance,
which is simply one minus the similarity, so a small distance means a close match.

## One rule you must not break

The documents and the questions must be embedded with the same model. Vectors from different models live in
different spaces and cannot be compared. If you change the embedding model, rebuild the whole database.

## Seeing embeddings

Vectors with hundreds of dimensions cannot be drawn, but a projection such as principal component analysis squeezes
them into two dimensions while keeping similar texts close together. On such a map, chunks about the same topic form
a cluster, and a question lands near the chunks that answer it.
