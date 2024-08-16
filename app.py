import gradio as gr
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import openai

# Set up OpenAI API key
openai.api_key = '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'

# Download NLTK data
nltk.download('punkt')
nltk.download('punkt_tab')

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
model = AutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

manual_path = "ubuntu_manual.txt"

# Load the Ubuntu manual from a .txt file
with open(manual_path, "r", encoding="utf-8") as file:
    full_text = file.read()

# Function to chunk the text into smaller pieces
def chunk_text(text, chunk_size=500):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []

    for sentence in sentences:
        if len(current_chunk) + len(sentence.split()) <= chunk_size:
            current_chunk.append(sentence)
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

# Apply chunking to the entire text
manual_chunks = chunk_text(full_text, chunk_size=500)

# Function to generate embeddings for each chunk
def embed_text(texts):
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # CLS token representation
    return embeddings

# Generate embeddings for the chunks
chunk_embeddings = embed_text(manual_chunks)

# Convert embeddings to a numpy array
chunk_embeddings_np = np.array(chunk_embeddings)

# Create a FAISS index and add the embeddings
dimension = chunk_embeddings_np.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings_np)

# Function to retrieve relevant chunks for a user query and print indices and distances
def retrieve_chunks(query, k=5):
    query_embedding = embed_text([query])
    distances, indices = index.search(query_embedding, k=k)
    valid_indices = [i for i in indices[0] if i < len(manual_chunks)]
    relevant_chunks = [manual_chunks[i] for i in valid_indices]
    
    # Print indices and distances
    for i, idx in enumerate(valid_indices):
        print(f"Index: {idx}, Distance: {distances[0][i]}")
    
    return relevant_chunks, indices[0], distances[0]

# Function to perform RAG: Retrieve chunks and generate a response using GPT-3.5
def rag_response_gpt3_5(query, k=3, max_tokens=150):
    relevant_chunks, indices, distances = retrieve_chunks(query, k=k)
    if not relevant_chunks:
        return "Sorry, I couldn't find relevant information."

    # Combine the query with a limited number of retrieved chunks
    augmented_input = query + "\n" + "\n".join(relevant_chunks)

    # Tokenize the augmented input and ensure it fits within model token limits
    input_ids = tokenizer(augmented_input, return_tensors="pt").input_ids[0]
    
    if len(input_ids) > 512:
        input_ids = input_ids[:512]
        augmented_input = tokenizer.decode(input_ids, skip_special_tokens=True)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": augmented_input}
        ],
        max_tokens=max_tokens,
        temperature=0.7
    )

    return response.choices[0].message['content'].strip()

# Chat history to maintain conversation context
def chatbot(query, history):
    if history is None:
        history = []

    # Retrieve relevant chunks along with their indices and distances
    relevant_chunks, indices, distances = retrieve_chunks(query)
    
    # Print the indices and distances of the retrieved chunks
    print(f"Retrieved Indices: {indices}")
    print(f"Retrieved Distances: {distances}")
    
    response = rag_response_gpt3_5(query)
    history.append((query, response))
    
    # Combine all messages into a single string
    chat_history = ""
    for user_input, bot_response in history:
        chat_history += f"User: {user_input}\nBot: {bot_response}\n\n"
    
    return chat_history, history

# Create the Gradio interface
iface = gr.Interface(fn=chatbot, 
                     inputs=["text", "state"], 
                     outputs=["text", "state"], 
                     title="Ubuntu Manual Chatbot",
                     description="Ask me anything about the Ubuntu manual.")

# Launch the app
if __name__ == "__main__":
    iface.launch()
