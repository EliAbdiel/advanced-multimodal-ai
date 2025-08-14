from .config import (
    DATABASE_URL,
    AZURE_CONTAINER_NAME,
    AZURE_STORAGE_ACCOUNT_NAME,
    AZURE_STORAGE_KEY
)
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient

storage_client = AzureBlobStorageClient(
    container_name=AZURE_CONTAINER_NAME,
    storage_account=AZURE_STORAGE_ACCOUNT_NAME,
    storage_key=AZURE_STORAGE_KEY
)

def get_persistent_data_layer():
    """Initializes the SQLAlchemy data layer for Chainlit."""
    # ALTER TABLE steps ADD COLUMN "defaultOpen" BOOLEAN DEFAULT false;
    # ALTER TABLE elements ADD COLUMN "autoPlay" BOOLEAN DEFAULT false;
    return SQLAlchemyDataLayer(
        conninfo=DATABASE_URL, 
        # ssl_require=True,
        storage_provider=storage_client,
    )