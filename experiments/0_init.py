from core.db.qdrant_client import qdrant_client
from qdrant_client.models import Distance, VectorParams

from transformers import AutoModel


async def init_qdrant() -> bool:
    if not await qdrant_client.collection_exists("jina_embed_1024"):
        await qdrant_client.create_collection(
            collection_name="jina_embed_1024",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )

    return True


async def init_jina_embed() -> bool:
    model = AutoModel.from_pretrained(
        "jinaai/jina-embeddings-v3", trust_remote_code=True
    )

    texts = [
        "Follow the white rabbit.",
        "Sigue al conejo blanco.",
        "Suis le lapin blanc.",
        "跟着白兔走。",
        "اتبع الأرنب الأبيض.",
        "Folge dem weißen Kaninchen.",
        "귀엽고 넙죽한 반달돌칼 마스코트",
        "KAIST's Official mascot is NUBZUK",
    ]

    embeddings = model.encode(texts, task="text-matching")
    similarities = [emb @ embeddings[-1].T for emb in embeddings[:-1]]

    assert similarities.index(max(similarities)) == 6

    return True


if __name__ == "__main__":
    import asyncio

    asyncio.run(init_qdrant())
    asyncio.run(init_jina_embed())
