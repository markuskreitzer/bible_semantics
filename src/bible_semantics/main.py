import asyncio
import json
import os
from pathlib import Path
import aiohttp
import pandas as pd
from tqdm import tqdm
import argparse

# Configuration for the model, endpoints, and batching.
MODEL = 'nomic-embed-text:latest'
ENDPOINTS = os.environ.get('AI_ENDPOINTS', "http://localhost:11434").split(",")
BATCH_SIZE = 100   # Number of texts per batch
RETRIES =  5       # Number of retry attempts for each batch
DELAY = 30          # Delay in seconds between retries

async def get_embeddings_batch(rows: list, session: aiohttp.ClientSession, endpoint: str) -> list:
    """
    Sends a POST request with a batch of texts (extracted from the rows) to get embeddings.
    Expects the API to accept a list of texts via the "input" field and return a JSON with a "data"
    list containing an embedding result for each text, in order.
    Returns a list of tuples: (row, embedding_result).
    """
    texts = [row["content"] for row in rows]
    url = f"{endpoint}/v1/embeddings"
    payload = {"model": MODEL, "input": texts}
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        data = await response.json()
        results = data.get("data", [])
        return [(row, result) for row, result in zip(rows, results)]

async def get_embeddings_batch_with_retry(rows: list, session: aiohttp.ClientSession, endpoint: str,
                                          retries: int = RETRIES, delay: int = DELAY) -> list:
    """
    Retry wrapper for get_embeddings_batch. It will attempt the request up to `retries` times,
    waiting `delay` seconds between attempts.
    """
    for attempt in range(1, retries + 1):
        try:
            return await get_embeddings_batch(rows, session, endpoint)
        except Exception as e:
            if attempt < retries:
                print(f"Batch failed on attempt {attempt}/{retries} for endpoint {endpoint}. "
                      f"Retrying in {delay} seconds. Error: {e}")
                await asyncio.sleep(delay)
            else:
                print(f"Batch failed on final attempt {attempt}/{retries} for endpoint {endpoint}. Error: {e}")
                raise

async def process_file(input_file: Path):
    """
    Processes a single JSONL file by:
      - Loading the file into a DataFrame.
      - Skipping rows that have been processed (based on a unique 'id' field).
      - Grouping rows into batches and sending embedding requests.
      - Appending the results to an output file named as the input file with _with_embeddings.
    """
    output_file = input_file.parent / f"{input_file.stem}_with_embeddings{input_file.suffix}"
    
    # Load the input JSONL file into a DataFrame.
    df = pd.read_json(input_file, lines=True)
    
    # Ensure there's an 'id' column (assign the index as id if missing).
    if "id" not in df.columns:
        df["id"] = df.index.astype(str)
    
    # Load processed IDs to enable resumability.
    processed_ids = set()
    if output_file.exists():
        with open(output_file, "r") as f:
            for line in f:
                try:
                    d = json.loads(line)
                    processed_ids.add(d["id"])
                except Exception as e:
                    print(f"Error reading a line in output file: {e}")
    
    # Filter out rows already processed.
    unprocessed_df = df[~df["id"].isin(processed_ids)]
    if unprocessed_df.empty:
        print(f"All rows in {input_file.name} have been processed.")
        return
    
    # Convert unprocessed rows to a list of dictionaries.
    rows = unprocessed_df.to_dict(orient="records")
    
    # Group rows into batches.
    batches = [rows[i: i+BATCH_SIZE] for i in range(0, len(rows), BATCH_SIZE)]
    
    tasks = []
    async with aiohttp.ClientSession() as session:
        # Create tasks; assign endpoints in a round-robin manner.
        for i, batch in enumerate(batches):
            endpoint = ENDPOINTS[i % len(ENDPOINTS)]
            tasks.append(asyncio.create_task(get_embeddings_batch_with_retry(batch, session, endpoint)))
        
        # Open output file in append mode.
        with open(output_file, "a") as f:
            # Use tqdm to track progress as batches complete.
            for batch_task in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                                     desc=f"Processing {input_file.name}"):
                try:
                    batch_results = await batch_task  # List of (row, embedding_result) tuples.
                    for row, embedding in batch_results:
                        # Store only the embedding vector (the list) from the API response.
                        row["embedding"] = embedding.get("embedding")
                        row["embedding_model"] = MODEL
                        f.write(json.dumps(row) + "\n")
                    f.flush()
                except Exception as e:
                    print(f"Error processing a batch in file {input_file.name}: {e}")

async def process_directory(data_dir: Path):
    """
    Processes all files in the given directory that match '*_flattened.jsonl'.
    """
    files = list(data_dir.glob("*_flattened.jsonl"))
    if not files:
        print(f"No files matching '*_flattened.jsonl' found in directory {data_dir}")
        return
    
    for file in files:
        print(f"\nProcessing file: {file.name}")
        await process_file(file)

def main():
    parser = argparse.ArgumentParser(
        description="Process a directory of JSONL files to add embedding vectors."
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing *_flattened.jsonl files")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    asyncio.run(process_directory(data_dir))

if __name__ == '__main__':
    main()

