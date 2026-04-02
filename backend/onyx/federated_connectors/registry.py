"""Registry mapping for federated connector classes."""

from pydantic import BaseModel

from onyx.configs.constants import FederatedConnectorSource


class FederatedConnectorMapping(BaseModel):
    module_path: str
    class_name: str


# Mapping of FederatedConnectorSource to connector details for lazy loading
FEDERATED_CONNECTOR_CLASS_MAP: dict[FederatedConnectorSource, FederatedConnectorMapping] = {}
