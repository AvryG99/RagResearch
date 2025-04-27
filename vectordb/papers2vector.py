import os
import uuid
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# === Load .env and keys ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, ".env"))

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
INDEX_NAME = "research-abstracts"

# === Initialize Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,  # Ensure this matches the embedding size
        metric='cosine',
        spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV)
    )

index = pc.Index(INDEX_NAME)

# === Load embedding model ===
model = SentenceTransformer("my_minilm_model")

# === Helper: build and upsert vectors from CSVs ===
def process_conference_year(folder_path, year):
    abstracts_path = os.path.join(folder_path, "abstracts.csv")
    authors_path = os.path.join(folder_path, "authors.csv")
    paper_info_path = os.path.join(folder_path, "paper_info.csv")

    if not (os.path.exists(abstracts_path) and os.path.exists(paper_info_path)):
        print(f"Missing CSVs in {folder_path}")
        return

    abstracts_df = pd.read_csv(abstracts_path, delimiter="~")
    authors_df = pd.read_csv(authors_path, delimiter="~")
    paper_info_df = pd.read_csv(paper_info_path, delimiter="~")
    print("Abstracts DataFrame:")
    print(abstracts_df.head())

    print("Authors DataFrame:")
    print(authors_df.head())

    print("Paper Info DataFrame:")
    print(paper_info_df.head())
    merged_df = abstracts_df.merge(authors_df, on="title", how="left") \
                            .merge(paper_info_df, on="title", how="left")
    
    print(merged_df)

    for _, row in merged_df.iterrows():
        title = str(row.get("title", "")).strip()
        abstract = str(row.get("abstract", "")).strip()

        if not title or len(abstract) < 30:
            continue

        # Generate the embedding for the abstract
        embedding = model.encode(abstract).tolist()
        vector_id = str(uuid.uuid4())

        metadata = {
            "title": title,
            "authors": str(row.get("authors", "")),
            "abstract_url": str(row.get("abstract_url", "")),
            "pdf_url": str(row.get("pdf_url", ""))
        }

        # Upsert the embedding and metadata into Pinecone
        try:
            upsert_response = index.upsert([{
                "id": vector_id,
                "values": embedding,
                "metadata": metadata
            }])
            print(f"Upsert response: {upsert_response}")
        except Exception as e:
            print(f"Error upserting vector for {title}: {e}")

    print(f"{year} papers indexed.")

# === Main entry ===
def build_vector_db(base_conference_path):
    for year in sorted(os.listdir(base_conference_path)):
        year_path = os.path.join(base_conference_path, year)
        if os.path.isdir(year_path):
            print(f"Processing CVPR {year}")
            process_conference_year(year_path, year)

    print("All papers indexed successfully.")

if __name__ == "__main__":
    paper_dir = os.path.join(BASE_DIR, "papers", "cvpr")
    build_vector_db(paper_dir)




# import os
# import uuid
# from tqdm import tqdm
# import fitz
# import re
# from sentence_transformers import SentenceTransformer
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec

# # === Load .env from parent directory and KEYS ===
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# load_dotenv(os.path.join(BASE_DIR, ".env"))
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# print(PINECONE_API_KEY)
# PINECONE_ENV = os.getenv("PINECONE_ENV") 
# INDEX_NAME = "research-sections"

# # === Section headers to extract ===
# SECTION_HEADERS = [
#     "abstract", "introduction", "related work", "background", "methodology",
#     "methods", "experiments", "results", "discussion", "conclusion", "references"
# ]

# # === Initialize Pinecone client ===
# pc = Pinecone(api_key=PINECONE_API_KEY)

# if INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=INDEX_NAME,
#         dimension=384,
#         metric='cosine',
#         spec=ServerlessSpec(
#             cloud="aws",
#             region=PINECONE_ENV
#         )
#     )

# index = pc.Index(INDEX_NAME)

# # === Load embedding model ===
# model = SentenceTransformer("my_minilm_model")

# # === PDF section extraction ===
# def extract_sections_from_pdf(filepath):
#     doc = fitz.open(filepath)
#     full_text = "\n".join(page.get_text() for page in doc)
#     doc.close()

#     full_text = full_text.lower()
#     sections = {}
#     current_section = None

#     lines = full_text.split('\n')
#     for line in lines:
#         header_match = re.match(r"^\s*(%s)\s*[:.]?\s*$" % "|".join(SECTION_HEADERS), line.strip())
#         if header_match:
#             current_section = header_match.group(1)
#             sections[current_section] = ""
#         elif current_section:
#             sections[current_section] += line + "\n"

#     return sections

# # === Main vectorization ===
# def build_vector_db(paper_dir):
#     for filename in tqdm(os.listdir(paper_dir)):
#         if not filename.endswith(".pdf"):
#             continue

#         file_path = os.path.join(paper_dir, filename)
#         paper_id = os.path.splitext(filename)[0]
#         sections = extract_sections_from_pdf(file_path)

#         for section, content in sections.items():
#             content = content.strip()
#             if len(content) < 100:
#                 continue  # Skip very short sections

#             embedding = model.encode(content).tolist()
#             uid = str(uuid.uuid4())

#             metadata = {
#                 "paper_id": paper_id,
#                 "section": section,
#                 "filename": filename
#             }

#             index.upsert([
#                 {
#                     "id": uid,
#                     "values": embedding,
#                     "metadata": metadata
#                 }
#             ])

#     print("✅ All papers indexed successfully.")


# if __name__ == "__main__":
#     paper_dir = os.path.join(BASE_DIR, "papers")
#     build_vector_db(paper_dir)

