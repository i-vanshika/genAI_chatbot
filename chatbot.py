import os
import numpy as np
import faiss
import pickle
from mistralai.client import MistralClient

client = MistralClient(api_key="bbBZCJFTcAamap3eWfsPr3VXG1rCFxLq")

def load_embeddings_and_index(embeddings_path, index_path):
    with open(embeddings_path, "rb") as f:
        embeddings = pickle.load(f)
    index = faiss.read_index(index_path)
    return embeddings, index

# Paths for loading files
data_dir = "data"
embeddings_path = os.path.join(data_dir, "embeddings.pkl")
index_path = os.path.join(data_dir, "faiss_index.index")

# Load embeddings and FAISS index
embeddings, index = load_embeddings_and_index(embeddings_path, index_path)
print("Loaded embeddings and FAISS index from disk.")

# Load the extracted text for chunk reference from multiple PDFs
all_texts = []
for filename in os.listdir(data_dir):
    if filename.endswith("_text.txt"):
        with open(os.path.join(data_dir, filename), "r", encoding="utf-8") as text_file:
            all_texts.append(text_file.read())

# Combine all texts into one string for chunk reference
combined_text = "\n".join(all_texts)

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Chunking the text
chunks = chunk_text(combined_text)

model = "codestral-latest"   ## LLM MODEL

def query_llm(prompt):
    response = client.completion(
        model=model,
        prompt=prompt,
        max_tokens=150
    )
    return str(response.choices[0].message).strip()

def get_response(query, threshold=0.75):
    query_embedding_response = client.embeddings(
        model="mistral-embed",
        input=[query]
    )
    query_embedding = np.array(query_embedding_response.data[0].embedding, dtype='float32').reshape(1, -1)

    # Search in the FAISS index
    D, I = index.search(query_embedding, k=5)

    # Filter out chunks based on distance (similarity) threshold
    filtered_indices = [i for i, d in zip(I[0], D[0]) if d < threshold]   
    if not filtered_indices:
        return "The information you're asking for is not found in the document."

    # Concatenate the filtered chunks into a context string
    context = "\n".join([chunks[i] for i in filtered_indices])
    prompt = f"Context: {context}\n\nQuestion: {query}\nAnswer:"
    response = query_llm(prompt)
    return response

# Get user input and generate response
user_query = input("Ask a question about the PDF content: ")

if user_query:
    answer = get_response(user_query, threshold= 0.6)  
    print("Answer:", answer)
    
