# Groq and Fast Inference

Groq is a company that runs large language models on its own hardware, the language processing unit. The service is
known for very high generation speed, often hundreds of tokens per second, which makes chat feel instant.

## Using the API

Groq offers an API that is compatible with the common chat completion format. You send a list of messages and the name
of a model, for example openai/gpt-oss-20b, and  you receive the answer. Streaming returns the answer piece by
piece, so the first words appear after a fraction of a second.

## Free tier and limits

A free key can be created in the Groq console. The free tier has rate limits that differ per model, such as a limited number of
requests and tokens per minute. When the limit is reached the API answers with a rate limit error and the application
should wait a moment and retry. When the servers are very busy the API can also return a temporary unavailable error.

## What Groq does in this project

Groq is used only for the last step. The embeddings are created locally, the search is done by ChromaDB, and Groq reads the retrieved
chunks and writes the final answer. A low temperature, such as 0.1, keeps the answer close to the provided text.

## Model choice

Larger models follow instructions and citations more reliably, while small instant models are faster and use less of the
daily quota. Model names change over time, so the application asks the API which models your key can use.
