import importlib.metadata


def get_cropioai_version() -> str:
    """Get the version number of CropioAI running the CLI"""
    return importlib.metadata.version("cropioai")
