import threading
import os

from core.logger import logger
from qdrant_client import AsyncQdrantClient
from pathlib import Path
from dotenv import load_dotenv


class Qdrant:
    _instance = None
    _initialized = False
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        cls._load_env(cls)
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def _load_env(self) -> None:
        env = os.getenv("PROFILE", "local")
        env_file = f".{env}.env"
        env_path = Path(__file__).parent.parent.parent / "resources" / env_file

        if env_path.exists():
            load_dotenv(dotenv_path=env_path)
        else:
            raise FileNotFoundError(f"Environment file not found: {env_path}")

    def __init__(self):
        if not self._initialized:
            self._client: AsyncQdrantClient = AsyncQdrantClient(
                url=os.environ.get("QDRANT_URL", ""),
                api_key=os.environ.get("QDRANT_API_KEY", None),
                prefer_grpc=False,
            )
            logger.info(
                "Qdrant client initialized",
            )

            self._initialized = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._client:
            await self._client.close()
            self._client = None

    @property
    def client(self) -> AsyncQdrantClient:
        return self._client


qdrant_client = Qdrant().client
