import asyncio
import uuid

import numpy as np
from typing import Any
from datasets import load_dataset
from qdrant_client.models import PointStruct

from core.db import qdrant_client
from core.logger import logger
from core.embedding import jina_client, JinaTask


COLLECTION_CONFIGS = {
    "jina_embed_1024": 1024,
    "jina_embed_512": 512,
    "jina_embed_128": 128,
}

QDRANT_BATCH_SIZE = 100


async def main() -> bool:
    try:
        wiki_qa_dataset = load_dataset("maywell/ko_wikidata_QA", split="train[:10%]")
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

    logger.info(f"Loaded {len(wiki_qa_dataset)} data points.")

    texts_for_batch_embedding: list[str] = []
    associated_metadata: list[dict[str, Any]] = []

    for i, item in enumerate(wiki_qa_dataset):
        instruction_text = item.get("instruction")
        output_text = item.get("output")

        if not output_text or not output_text.strip():
            continue
        if not instruction_text or not instruction_text.strip():
            continue

        texts_for_batch_embedding.append(output_text)
        associated_metadata.append(
            {
                "id": str(uuid.uuid4()),
                "instruction": instruction_text,
                "output": output_text,
            }
        )

    if not texts_for_batch_embedding:
        logger.error("No valid texts found for embedding.")
        return False

    try:
        loop = asyncio.get_event_loop()
        task_for_embedding = JinaTask.RETRIEVAL_PASSAGE

        full_embeddings_list = await loop.run_in_executor(
            None, jina_client.encode, texts_for_batch_embedding, task_for_embedding
        )
    except Exception as e:
        logger.error(f"Error during batch Jina embedding: {e}")
        return False

    if len(full_embeddings_list) != len(texts_for_batch_embedding):
        logger.error(
            f"Error: Mismatch between number of texts ({len(texts_for_batch_embedding)}) "
            f"and number of embeddings returned ({len(full_embeddings_list)})."
        )
        return False

    points_to_upsert_batched: dict[str, list[PointStruct]] = {
        name: [] for name in COLLECTION_CONFIGS.keys()
    }

    processed_documents_count = 0

    for i, full_embedding in enumerate(full_embeddings_list):
        metadata = associated_metadata[i]
        doc_id = metadata["id"]
        instruction_text = metadata["instruction"]

        if (
            not isinstance(full_embedding, (list, np.ndarray))
            or len(full_embedding) == 0
        ):
            logger.warning(
                f"Warning: Invalid or empty embedding received for item with instruction: '{instruction_text[:50]}...'. Skipping."
            )
            continue

        payload = {"ground_truth": instruction_text}

        for collection_name, target_dim in COLLECTION_CONFIGS.items():
            if len(full_embedding) < target_dim:
                logger.error(
                    f"Error: Base embedding (padded or not) is too short ({len(full_embedding)}d) for target dimension {target_dim} in {collection_name}. Skipping point."
                )
                continue

            current_embedding = full_embedding[:target_dim]

            points_to_upsert_batched[collection_name].append(
                PointStruct(id=doc_id, vector=current_embedding, payload=payload)
            )

        processed_documents_count += 1

        if (
            processed_documents_count % QDRANT_BATCH_SIZE == 0
            or i == len(full_embeddings_list) - 1
        ):
            for col_name, points_list in points_to_upsert_batched.items():
                if points_list:
                    logger.info(
                        f"Upserting batch of {len(points_list)} points to {col_name} (Total docs processed: {processed_documents_count})..."
                    )
                    try:
                        await qdrant_client.upsert(
                            collection_name=col_name, points=points_list, wait=True
                        )
                    except Exception as e:
                        logger.error(f"Error upserting batch to {col_name}: {e}")
                    points_to_upsert_batched[col_name] = []

    return True


if __name__ == "__main__":
    asyncio.run(main())
