import asyncio
import argparse
from document_upload import DocumentUploader
import os
from dotenv import load_dotenv

"""
This script is used to upload documents to a MongoDB database with embeddings.
It supports various file types and can process directories recursively.

# Basic usage with default settings
python upload_run.py -d /path/to/your/documents

# Specify custom collection and database names
python upload_run.py -d /path/to/your/documents -c my_collection -db my_database

# Process subdirectories recursively
python upload_run.py -d /path/to/your/documents -r

# Use a different embedding model
python upload_run.py -d /path/to/your/documents -m my_model
"""

async def main():
    # Load environment variables
    load_dotenv()
    
    # Get MongoDB connection string from environment variable
    mongo_connection = os.getenv('MONGODB_CONNECTION_STRING')
    if not mongo_connection:
        raise ValueError("MONGODB_CONNECTION_STRING environment variable is not set")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Upload documents to MongoDB with embeddings')
    parser.add_argument('--directory', '-d', type=str, required=True,
                       help='Directory path containing documents to process')
    parser.add_argument('--collection', '-c', type=str, default='documents',
                       help='MongoDB collection name (default: documents)')
    parser.add_argument('--database', '-db', type=str, default='semantic_rag',
                       help='MongoDB database name (default: semantic_rag)')
    parser.add_argument('--model', '-m', type=str, default='nomic-embed-text',
                       help='Ollama model name (default: nomic-embed-text)')
    parser.add_argument('--recursive', '-r', action='store_true',
                       help='Process subdirectories recursively')

    args = parser.parse_args()

    # Initialize document uploader
    uploader = DocumentUploader(
        connection_string=mongo_connection,
        model_name=args.model
    )

    print(f"Processing documents in: {args.directory}")
    print(f"Using database: {args.database}")
    print(f"Using collection: {args.collection}")
    print(f"Recursive mode: {args.recursive}")

    try:
        # Process the directory
        results = await uploader.process_input(
            input_path=args.directory,
            collection_name=args.collection,
            db_name=args.database,
            recursive=args.recursive
        )

        # Print results
        print("\nProcessing Results:")
        print(f"Total chunks processed: {results['total_chunks']}")
        print(f"Successfully processed files: {len(results['successful'])}")
        print(f"Failed files: {len(results['failed'])}")
        print(f"Skipped files: {len(results['skipped'])}")

        if results['failed']:
            print("\nFailed files:")
            for fail in results['failed']:
                print(f"- {fail['filename']}: {fail['message']}")

        if results['skipped']:
            print("\nSkipped files:")
            for skip in results['skipped']:
                print(f"- {skip}")

    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())