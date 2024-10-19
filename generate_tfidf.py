import os
import fitz  # PyMuPDF
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

# Set your document directory
docs_path = r"C:\Users\umesh\Project Documents"
pdf_files = [f for f in os.listdir(docs_path) if f.endswith('.pdf')]

# Initialize text storage
documents = []

# Download NLTK resources
nltk.download('punkt')

# Function to extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

# Extract text from each PDF and preprocess
for pdf_file in pdf_files:
    pdf_path = os.path.join(docs_path, pdf_file)
    extracted_text = extract_text_from_pdf(pdf_path)
    documents.append(extracted_text)

# Initialize the TF-IDF vectorizer and fit it to the documents
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(documents)

# Save the TF-IDF vectorizer, TF-IDF matrix, and document names
with open(os.path.join(docs_path, 'tfidf_vectorizer.pkl'), 'wb') as f:
    pickle.dump(vectorizer, f)

with open(os.path.join(docs_path, 'tfidf_matrix.pkl'), 'wb') as f:
    pickle.dump(tfidf_matrix, f)

# Save document names
doc_names = [f"Document {i + 1}: {pdf_file}" for i, pdf_file in enumerate(pdf_files)]
with open(os.path.join(docs_path, 'doc_names.pkl'), 'wb') as f:
    pickle.dump(doc_names, f)

print("TF-IDF files generated successfully!")
