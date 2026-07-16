from app.config.config import Config
# final from single window 

import os
import json
import logging
import concurrent.futures
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from tqdm import tqdm
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Set up logging
logging.basicConfig(
    filename='/home/abusufyan/development/private_llm/app/embeddings/pipeline.log', 
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

def get_pdf_file_paths(root_directory):
    pdf_files_to_process = []
    for root, dirs, files in os.walk(root_directory):
        pdf_files_to_process.extend([os.path.join(root, file) for file in files if file.lower().endswith(".pdf")])
    pdf_files_to_process.sort()  # Sort the list of file paths
    logger.info(f"Found {len(pdf_files_to_process)} PDF files to process.")
    return pdf_files_to_process

def process_pdf(pdf_file_path):
    try:
        pdf_loader = PyPDFLoader(pdf_file_path)
        docs = pdf_loader.load()
        logger.info(f"Successfully loaded file: {pdf_file_path}")
        return docs
    except Exception as e:
        logger.error(f"Error processing file {pdf_file_path}: {e}")
        return []

def split_documents(docs, chunk_size=1500, chunk_overlap=300):
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, 
            chunk_overlap=chunk_overlap
            )
        splits = text_splitter.split_documents(docs)
        logger.info(f"Successfully split documents into {len(splits)} chunks.")
        return splits
    except Exception as e:
        logger.error(f"Error splitting documents: {e}")
        return []

def create_embeddings_and_store(docs, embedding, index_name, namespace):
    try:
        from langchain_pinecone import PineconeVectorStore

        PineconeVectorStore.from_documents(
            documents=docs,
            embedding=embedding,
            index_name=index_name,
            namespace=namespace,
        )
        logger.info(f"Successfully created embeddings and stored in Pinecone index {index_name}.")
    except Exception as e:
        logger.error(f"Error creating embeddings: {e}")

def process_pdf_batch(pdf_file, embedding, index_name, namespace, progress_file):
    try:
        docs = process_pdf(pdf_file)
        splits = split_documents(docs)
        create_embeddings_and_store(splits, embedding, index_name, namespace)
        save_progress(progress_file, pdf_file)
    except Exception as e:
        logger.error(f"Error processing file {pdf_file}: {e}")

def save_progress(progress_file, last_processed_file):
    try:
        with open(progress_file, 'w') as f:
            json.dump({'last_processed_file': last_processed_file}, f)
        logger.info(f"Progress saved. Last processed file: {last_processed_file}")
    except Exception as e:
        logger.error(f"Error saving progress: {e}")

def load_progress(progress_file):
    try:
        if os.path.exists(progress_file):
            with open(progress_file, 'r') as f:
                progress_data = json.load(f)
                last_processed_file = progress_data.get('last_processed_file')
                logger.info(f"Loaded progress. Last processed file: {last_processed_file}")
                return last_processed_file
        return None
    except Exception as e:
        logger.error(f"Error loading progress: {e}")
        return None

def process_pdfs_in_batches(pdf_files_to_process, batch_size, embedding, index_name, namespace, progress_file='progress.json'):
    # Load progress to resume from the last processed file
    last_processed_file = load_progress(progress_file)
    if last_processed_file:
        last_index = pdf_files_to_process.index(last_processed_file) + 1
        pdf_files_to_process = pdf_files_to_process[last_index:]
        initial_progress = last_index
    else:
        initial_progress = 0

    total_files = len(pdf_files_to_process) + initial_progress

    with tqdm(total=total_files, initial=initial_progress, desc=f"{Fore.GREEN}Processing files", unit="file") as pbar:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            future_to_pdf = {executor.submit(process_pdf_batch, pdf_file, embedding, index_name, namespace, progress_file): pdf_file for pdf_file in pdf_files_to_process}

            for future in concurrent.futures.as_completed(future_to_pdf):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    logger.error(f"Error processing file: {e}")

def main():
    root_directory = "/home/abusufyan/development/private_llm/app/embeddings/data"
    index_name = Config.PINECONE_INDEX_NAME
    namespace = Config.PINECONE_NAMESPACE
    batch_size = 1
    progress_file = '/home/abusufyan/development/private_llm/app/embeddings/progress.json'

    # Initialize embedding
    embedding = OpenAIEmbeddings(model=Config.OPENAI_EMBEDDING_MODEL, api_key=Config.OPENAI_API_KEY)

    # Get PDF file paths
    pdf_files_to_process = get_pdf_file_paths(root_directory)

    # Print the list of sorted PDF file paths
    print("List of sorted PDF file paths:")
    for path in pdf_files_to_process:
        print(path)

    # Process PDFs in batches
    process_pdfs_in_batches(pdf_files_to_process, batch_size, embedding, index_name, namespace, progress_file)

    # Print the final message in the middle of the console output
    final_message = Fore.GREEN + "All PDF files have been Embedded and stored in VectorStore."
    print("\n" * (os.get_terminal_size().lines // 2))
    print(final_message.center(os.get_terminal_size().columns))
    logger.info("All PDF files have been Embedded and stored in VectorStore.")

if __name__ == "__main__":
    main()
