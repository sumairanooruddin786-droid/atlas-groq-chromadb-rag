# Vector Databases and ChromaDB

A vector database stores embeddings together with the original text and metadata, and it can quickly find the vectors
nearest to a query vector. Regular databases search by exact values; a vector database searches by meaning.

## ChromaDB

ChromaDB is an open source vector database that runs inside your Python program and saves its data in a local folder.
Every record has an id, an embedding, the text of the chunk and metadata such as the file name, page number and chunk
number. Records are written with upsert, which inserts a new record or updates an existing one with the same id.

## Approximate nearest neighbour search

Comparing a query with every stored vector would be slow for millions of chunks, so ChromaDB builds an HNSW graph index.
The index finds very close neighbours quickly instead of checking everything. The distance function is chosen when the
collection is created; this project uses cosine distance.

## Metadata filters

A search can be restricted with a filter. For example, asking only the file policy.pdf limits the results to chunks whose
source metadata equals that file name. This is how a user can ask a question about one document out of many.

## Persistence

Because ChromaDB is persistent, the index survives restarts. Delete the database folder, or call the reset function,
to start again from an empty collection.
