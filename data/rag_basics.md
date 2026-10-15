# Retrieval-Augmented Generation (RAG)

Retrieval-Augmented Generation, or RAG, is a pattern that lets a language model answer questions using documents
that it was never trained on. Instead of relying on memory, the system looks up the relevant text first and gives it
to the model together with the question.

## The three steps

1. **Retrieve.** The question is converted into a vector and compared with the vectors of stored text chunks. The most
   similar chunks are selected.
2. **Augment.** The selected chunks are placed into the prompt as numbered excerpts.
3. **Generate.** The language model writes an answer that uses only those excerpts and cites them.

## Why RAG is useful

A plain language model can hallucinate. It may state a policy, a number or a date that sounds right but is invented.
RAG reduces this because the answer is built from real text, and the user can open the cited chunk to verify the claim.
It also keeps knowledge fresh: to update what the assistant knows, you replace a document instead of retraining a model.

## Chunking

Long documents are split into chunks of roughly one thousand characters with a small overlap, so a sentence at the
edge of one chunk is still available in the next one. Chunks that are too small lose context, and chunks that are
too large dilute the meaning and waste the prompt budget.

## When the answer is missing

A well designed RAG assistant must be able to say that it does not know. If no chunk is similar enough to the
question, the assistant should answer that the information is not in the supplied documents rather than guessing.
