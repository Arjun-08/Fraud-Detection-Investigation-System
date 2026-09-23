# RAG Methodology
RAG separates retrieval from generation. A query q becomes an embedding e_q and a knowledge chunk d_i becomes e_i. With normalized embeddings, similarity can be represented as sim(q,d_i) = e_q^T e_i. The top-k chunks are supplied to the language model as evidence.
