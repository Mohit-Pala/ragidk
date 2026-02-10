print("staritng")
import os
print("imported os")
import torch
print("imported torch")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("imported transformers")
from sentence_transformers import SentenceTransformer, CrossEncoder
print("imported sentence_transformers")
from rank_bm25 import BM25Okapi
print("imported rank_bm25")
import numpy as np
print("imported numpy")
import fitz
print("imported fitz")

# clear screen
os.system('clear')

# configs
data_dir = 'data/'
rag_output_dir = 'rags/'

MODEL_ID = "google/gemma-3-4b-it"
EMBED_MODEL_ID = "all-MiniLM-L6-v2"  # Fast embedding model
RERANK_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Re-ranking model


# train methods
def process_pdf_with_metadata(path):
    doc = fitz.open(path)
    chunks_with_metadata = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text: continue
        words = text.split()
        for i in range(0, len(words), 100):  # More overlap: 100 word stride
            chunk_text = " ".join(words[i : i + 200]) # 100 word overlap now
            chunks_with_metadata.append({
                "text": chunk_text,
                "page": page_num + 1  # 1-indexed for humans
            })
    return chunks_with_metadata

def get_embeddings(data, embed_model):
    """Use sentence-transformers for faster, better embeddings"""
    texts = [item["text"] for item in data]
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_model.encode(
        texts, 
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=32
    )
    return embeddings





# run methods


@torch.no_grad()
def answer_question(question, tokenizer, model, chunk_embeddings, chunks_data, embed_model, bm25, reranker, top_k=10):
    """Hybrid search with BM25 + semantic + re-ranking"""
    
    # 1. Semantic search
    q_embed = embed_model.encode(question, convert_to_tensor=True)
    semantic_scores = torch.nn.functional.cosine_similarity(
        q_embed.unsqueeze(0), 
        chunk_embeddings
    ).cpu().numpy()
    
    # 2. BM25 keyword search
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    
    # 3. Normalize and combine scores (0.4 BM25 + 0.6 semantic)
    semantic_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.max() - semantic_scores.min() + 1e-8)
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min() + 1e-8)
    hybrid_scores = 0.4 * bm25_norm + 0.6 * semantic_norm
    
    # 4. Get top candidates for re-ranking
    top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
    
    # 5. Re-rank with cross-encoder
    pairs = [[question, chunks_data[i]["text"]] for i in top_indices]
    rerank_scores = reranker.predict(pairs)
    reranked_indices = np.argsort(rerank_scores)[::-1]
    
    # Get top 5 after re-ranking
    final_top_k = min(5, len(reranked_indices))
    final_indices = [top_indices[i] for i in reranked_indices[:final_top_k]]
    
    # Combine context from top chunks
    combined_context = ""
    page_refs = set()
    for i, idx in enumerate(final_indices):
        chunk = chunks_data[idx]
        combined_context += f"\n[Chunk {i+1}, Page {chunk['page']}]\n{chunk['text']}\n"
        page_refs.add(chunk['page'])
    
    page_str = ", ".join(map(str, sorted(page_refs)))
    
    messages = [
        {
            "role": "user", 
            "content": (
                f"Use the provided context from Pages {page_str} to answer the question. "
                f"IMPORTANT: You must include at least one direct quote from the text in your answer "
                f"and cite it with the page number (Source: Page X).\n\n"
                f"CONTEXT (Top {final_top_k} most relevant chunks):\n{combined_context}\n\n"
                f"QUESTION: {question}"
            )
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    
    result = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response, [chunks_data[i] for i in final_indices], page_str











def train():
    # get all pdf files from data_dir
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    print('which pdf to create embeddings for?')
    for i, pdf_file in enumerate(pdf_files):
        print(f"{i} -> {pdf_file}")
    choice = int(input("->"))
    try:
        pdf_file = pdf_files[choice]
    except IndexError:
        print("bruh")
        os._exit(0)
    
    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    
    print("Processing PDF...")
    data_list = process_pdf_with_metadata(os.path.join(data_dir, pdf_file))
    print(f"Created {len(data_list)} chunks")
    
    print("Generating embeddings...")
    vecs = get_embeddings(data_list, embed_model)
    
    # Create BM25 index
    print("Creating BM25 index...")
    corpus = [chunk["text"].lower().split() for chunk in data_list]
    bm25 = BM25Okapi(corpus)
    
    save_path = os.path.join(rag_output_dir, pdf_file.replace('.pdf', '.pt'))
    torch.save({"embeddings": vecs, "data": data_list, "bm25_corpus": corpus}, save_path)
    print(f"Saved to {save_path}")
    


def run():
    # get all .pt files
    pt_files = [f for f in os.listdir(rag_output_dir) if f.endswith('.pt')]
    print('which rag to laod?')
    for i, pt_file in enumerate(pt_files):
        print(f"{i} -> {pt_file}")
    choice = int(input("->"))
    try:
        pt_file = pt_files[choice]
    except IndexError:
        print("bruh")
        os._exit(0)
    
    DB_PATH = os.path.join(rag_output_dir, pt_file)
    
    print("Loading data...")
    data = torch.load(DB_PATH, map_location="cpu")
    chunks_data = data['data']
    chunk_embeddings = data['embeddings'].to("cuda", dtype=torch.float32)
    
    print("Loading BM25 index...")
    if 'bm25_corpus' in data:
        bm25 = BM25Okapi(data['bm25_corpus'])
    else:
        corpus = [chunk["text"].lower().split() for chunk in chunks_data]
        bm25 = BM25Okapi(corpus)
    
    print("Loading models...")
    embed_model = SentenceTransformer(EMBED_MODEL_ID)
    reranker = CrossEncoder(RERANK_MODEL_ID)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    os.system('clear')
    
    while True:
        q = input("\nQues, q to quit: ")
        if q.lower() in ['exit', 'quit', 'q']: break
        
        ans, top_chunks, pg = answer_question(q, tokenizer, model, chunk_embeddings, chunks_data, embed_model, bm25, reranker)
        print(f"\nAns: {ans}")
        print(f"\nSauce: {pg}")
        print(f"\nTop chunks used:")
        for i, chunk in enumerate(top_chunks[:3]):
            print(f"  [{i+1}] Page {chunk['page']}: {chunk['text'][:100]}...")
        
        for i in range(3):
            print("\n")


print("1 -> train \n2 -> run\nq => quit")
mode = input("->").strip()
if mode.lower() in ('q', 'quit', 'exit', ''):
    print("Exiting.")
    os._exit(0)
if mode == '1':
    train()
elif mode == '2':
    run()
else:
    print('bruh')
