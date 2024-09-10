import os
import fitz  # PyMuPDF
import numpy as np
import faiss
import pickle
from mistralai.client import MistralClient

client = MistralClient(api_key="bbBZCJFTcAamap3eWfsPr3VXG1rCFxLq")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def chunk_text(text, chunk_size=512):
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def split_into_batches(chunks, max_tokens=16384):
    batches = []
    current_batch = []
    current_batch_size = 0

    for chunk in chunks:
        chunk_size = len(chunk.split())
        if current_batch_size + chunk_size > max_tokens:
            if current_batch:  # Ensure the current batch is not empty
                batches.append(current_batch)
            current_batch = [chunk]
            current_batch_size = chunk_size
        else:
            current_batch.append(chunk)
            current_batch_size += chunk_size
    
    if current_batch:
        batches.append(current_batch)
    
    return batches

def get_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        chunk_size = len(chunk.split())
        if chunk_size > 16384:
            # If chunk exceeds limit, split into smaller chunks
            smaller_chunks = chunk_text(chunk, chunk_size=4096)
            for small_chunk in smaller_chunks:
                embeddings_batch_response = client.embeddings(
                    model="mistral-embed",
                    input=[small_chunk],
                )
                embeddings.extend([item.embedding for item in embeddings_batch_response.data])
        else:
            embeddings_batch_response = client.embeddings(
                model="mistral-embed",
                input=[chunk],
            )
            embeddings.extend([item.embedding for item in embeddings_batch_response.data])
    
    return embeddings

def save_embeddings_and_index(embeddings, index, embeddings_path, index_path):
    with open(embeddings_path, "wb") as f:
        pickle.dump(embeddings, f)
    faiss.write_index(index, index_path)

def create_faiss_index(embeddings):
    embeddings_np = np.array(embeddings, dtype='float32')
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_np)
    return index

# Create directory for saving files
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# Paths for saving files
embeddings_path = os.path.join(data_dir, "embeddings.pkl")
index_path = os.path.join(data_dir, "faiss_index.index")
pdf_paths = ["howto-unicode.pdf", "howto-sorting.pdf", "howto-descriptor.pdf"]

# Extract and save text from multiple PDFs
all_texts = []
for pdf_path in pdf_paths:
    text = extract_text_from_pdf(pdf_path)
    all_texts.append(text)
    with open(os.path.join(data_dir, f"{os.path.splitext(os.path.basename(pdf_path))[0]}_text.txt"), "w", encoding="utf-8") as text_file:
        text_file.write(text)

# Combine all texts into one string
combined_text = "\n".join(all_texts)

# Chunk combined text
chunks = chunk_text(combined_text)

# Generate embeddings and create FAISS index
embeddings = get_embeddings(chunks)
index = create_faiss_index(embeddings)

# Save embeddings and FAISS index
save_embeddings_and_index(embeddings, index, embeddings_path, index_path)
print("Preprocessing complete and data saved.")