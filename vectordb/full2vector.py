import os
import uuid
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec

# === Load environment variables ===
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "paper-contents"

# === Initialize Pinecone Client ===
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

# === Load embedding model ===
model = SentenceTransformer("my_minilm_model")

# === Load the paper data and split it into chunks ===
def split_text_into_chunks(text, chunk_size=1000, overlap_size=200):
    """Split text into overlapping chunks based on the token count."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

def extract_text_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = ""
    for page in doc:
        full_text += page.get_text("text") 
    doc.close()
    return full_text

# === Main processing function to upsert into Pinecone ===
def build_vector_db(base_paper_dir):
    years = ["2022", "2023", "2024"]
    papers_for_review = []

    for year in years:
        print(f"📚 Processing year {year}...")
        year_paper_dir = os.path.join(base_paper_dir, "cvpr", year, "papers")
        authors_csv_path = os.path.join(base_paper_dir, "cvpr", year, "authors.csv")

        if not os.path.exists(authors_csv_path):
            print(f"⚠️ Warning: authors.csv for year {year} not found. Skipping.")
            continue
        authors_df = pd.read_csv(authors_csv_path, delimiter="~")
        titles = authors_df['title'].tolist()

        pdf_files = sorted([f for f in os.listdir(year_paper_dir) if f.endswith(".pdf")], key=lambda x: int(os.path.splitext(x)[0]))

        for pdf_file in tqdm(pdf_files, desc=f"Year {year}"):
            file_path = os.path.join(year_paper_dir, pdf_file)
            paper_id = os.path.splitext(pdf_file)[0]
            paper_index = int(paper_id)

            if paper_index >= len(titles):
                print(f"⚠️ Warning: No title for paper {pdf_file}. Skipping.")
                continue

            paper_title = titles[paper_index]
            full_text = extract_text_from_pdf(file_path)
            chunks = split_text_into_chunks(full_text)

            for chunk in chunks:
                chunk = chunk.strip()
                if len(chunk) < 100:
                    continue

                embedding = model.encode(chunk).tolist()
                uid = str(uuid.uuid4())

                papers_for_review.append({
                    "id": uid,
                    "values": embedding,
                    "metadata": {
                        "paper_id": f"{year}_{paper_id}",
                        "title": paper_title,
                        "year": year,
                        "content": chunk
                    }
                })

    # === Upsert papers in small batches ===
    batch_size = 50  # Adjust depending on vector size
    for i in range(0, len(papers_for_review), batch_size):
        batch = papers_for_review[i:i+batch_size]
        index.upsert(vectors=batch)

    print(f"✅ All papers processed and upserted to Pinecone.")

if __name__ == "__main__":
    paper_dir = "/home/averyg99/RagResearch/papers"
    build_vector_db(paper_dir)
