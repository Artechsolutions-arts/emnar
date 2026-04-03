

CONNECTOR_VALIDATION_ERROR_MESSAGE_PREFIX = "ConnectorValidationError:"


# Sentinel value to distinguish between "not provided" and "explicitly set to None"
class UnsetType:
    def __repr__(self) -> str:
        return "<UNSET>"


UNSET = UnsetType()
