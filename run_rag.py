print("staritng")
import os
print("imported os")
import torch
print("imported torch")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("imported transformers")
import fitz
print("imported fitz")

# clear screen
os.system('clear')

# configs
data_dir = 'data/'
rag_output_dir = 'rags/'

MODEL_ID = "google/gemma-3-4b-it"


# train methods
def process_pdf_with_metadata(path):
    doc = fitz.open(path)
    chunks_with_metadata = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if not text: continue
        words = text.split()
        for i in range(0, len(words), 150):  # ~200 words per chunk
            chunk_text = " ".join(words[i : i + 200]) # 50 word overlap
            chunks_with_metadata.append({
                "text": chunk_text,
                "page": page_num + 1  # 1-indexed for humans
            })
    return chunks_with_metadata

@torch.no_grad()
def get_embeddings(data, tokenizer, model):
    embeddings = []
    for i, item in enumerate(data):
        inputs = tokenizer(item["text"], return_tensors="pt", truncation=True, max_length=512).to("cuda")
        outputs = model.model(inputs.input_ids, output_hidden_states=True)
        embed = outputs.hidden_states[-1].mean(dim=1).cpu()
        embeddings.append(embed)
        if i % 100 == 0: print(f"Progress: {i}/{len(data)}")
    return torch.cat(embeddings)





# run methods


@torch.no_grad()
def answer_question(question, tokenizer, model, chunk_embeddings, chunks_data):
    q_inputs = tokenizer(question, return_tensors="pt").to("cuda")
    q_outputs = model.model(q_inputs.input_ids, output_hidden_states=True)
    q_embed = q_outputs.hidden_states[-1].mean(dim=1)

    similarities = torch.nn.functional.cosine_similarity(q_embed, chunk_embeddings)
    top_idx = torch.argmax(similarities).item()
    
    source_text = chunks_data[top_idx]["text"]
    page_num = chunks_data[top_idx]["page"]

    messages = [
        {
            "role": "user", 
            "content": (
                f"Use the provided context from Page {page_num} to answer the question. "
                f"IMPORTANT: You must include at least one direct quote from the text in your answer "
                f"and cite it as (Source: Page {page_num}).\n\n"
                f"CONTEXT:\n{source_text}\n\n"
                f"QUESTION: {question}"
            )
        }
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    
    result = model.generate(**inputs, max_new_tokens=400, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    response = tokenizer.decode(result[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    return response, source_text, page_num











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
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    
    data_list = process_pdf_with_metadata(os.path.join(data_dir, pdf_file))
    
    vecs = get_embeddings(data_list, tokenizer, model)
    save_path = os.path.join(rag_output_dir, pdf_file.replace('.pdf', '.pt'))
    torch.save({"embeddings": vecs, "data": data_list}, save_path)
    


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
    
    data = torch.load(DB_PATH, map_location="cpu")
    chunks_data = data['data']
    chunk_embeddings = data['embeddings'].to("cuda", dtype=torch.bfloat16)
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto")
    os.system('clear')
    while True:
        q = input("\nQues, q to quit: ")
        if q.lower() in ['exit', 'quit', 'q']: break
        
        ans, raw, pg = answer_question(q, tokenizer, model, chunk_embeddings, chunks_data)
        print(f"\nAns: {ans}")
        print(f"\nSauce: {pg})\n")
        print(f"[{raw[:200]}")
        
        for i in range(5):
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
