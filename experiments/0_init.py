from core.db import qdrant_client
from core.embedding import jina_client, JinaTask
from qdrant_client.models import Distance, VectorParams


async def init_qdrant() -> bool:
    if not await qdrant_client.collection_exists("jina_embed_1024"):
        await qdrant_client.create_collection(
            collection_name="jina_embed_1024",
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    
    if not await qdrant_client.collection_exists("jina_embed_512"):
        await qdrant_client.create_collection(
            collection_name="jina_embed_512",
            vectors_config=VectorParams(size=512, distance=Distance.COSINE),
        )
    
    if not await qdrant_client.collection_exists("jina_embed_128"):
        await qdrant_client.create_collection(
            collection_name="jina_embed_128",
            vectors_config=VectorParams(size=128, distance=Distance.COSINE),
        )

    return True


async def init_jina_embed() -> bool:
    texts = [
        "Follow the white rabbit.",
        "Sigue al conejo blanco.",
        "Suis le lapin blanc.",
        "跟着白兔走。",
        "اتبع الأرنب الأبيض.",
        "Folge dem weißen Kaninchen.",
        "KAIST's Official mascot is NUBZUK",
    ]

    query = "귀엽고 넙죽한 반달돌칼 마스코트"

    result = jina_client.find_similar_texts(
        query,
        texts,
        task=[JinaTask.RETRIEVAL_QUERY, JinaTask.RETRIEVAL_PASSAGE],
        matryoshka_dim=512,
    )
    print(result)

    assert result["index"][0] == 6

    return True


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_qdrant())
    asyncio.run(init_jina_embed())
