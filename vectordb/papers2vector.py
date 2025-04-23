import os
import uuid
from tqdm import tqdm
import fitz
import re
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# === Load .env from parent directory and KEYS ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
print(PINECONE_API_KEY)
PINECONE_ENV = os.getenv("PINECONE_ENV") 
INDEX_NAME = "research-sections"

# === Section headers to extract ===
SECTION_HEADERS = [
    "abstract", "introduction", "related work", "background", "methodology",
    "methods", "experiments", "results", "discussion", "conclusion", "references"
]

# === Initialize Pinecone client ===
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud="aws",
            region=PINECONE_ENV
        )
    )

index = pc.Index(INDEX_NAME)

# === Load embedding model ===
model = SentenceTransformer("my_minilm_model")

# === PDF section extraction ===
def extract_sections_from_pdf(filepath):
    doc = fitz.open(filepath)
    full_text = "\n".join(page.get_text() for page in doc)
    doc.close()

    full_text = full_text.lower()
    sections = {}
    current_section = None

    lines = full_text.split('\n')
    for line in lines:
        header_match = re.match(r"^\s*(%s)\s*[:.]?\s*$" % "|".join(SECTION_HEADERS), line.strip())
        if header_match:
            current_section = header_match.group(1)
            sections[current_section] = ""
        elif current_section:
            sections[current_section] += line + "\n"

    return sections

# === Main vectorization ===
def build_vector_db(paper_dir):
    for filename in tqdm(os.listdir(paper_dir)):
        if not filename.endswith(".pdf"):
            continue

        file_path = os.path.join(paper_dir, filename)
        paper_id = os.path.splitext(filename)[0]
        sections = extract_sections_from_pdf(file_path)

        for section, content in sections.items():
            content = content.strip()
            if len(content) < 100:
                continue  # Skip very short sections

            embedding = model.encode(content).tolist()
            uid = str(uuid.uuid4())

            metadata = {
                "paper_id": paper_id,
                "section": section,
                "filename": filename
            }

            index.upsert([
                {
                    "id": uid,
                    "values": embedding,
                    "metadata": metadata
                }
            ])

    print("✅ All papers indexed successfully.")


if __name__ == "__main__":
    paper_dir = os.path.join(BASE_DIR, "papers")
    build_vector_db(paper_dir)
