from core.logger import logger
from core.db.qdrant_client import qdrant_client
from qdrant_client.models import Distance, VectorParams


async def init_qdrant():
    if not await qdrant_client.collection_exists("nomic_embed"):
        await qdrant_client.create_collection(
            collection_name="my_collection",
            vectors_config=VectorParams(size=100, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection 'nomic_embed' created")


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_qdrant())
