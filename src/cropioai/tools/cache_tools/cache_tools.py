from pydantic import BaseModel, Field

from cropioai.agents.cache import CacheHandler
from cropioai.tools.structured_tool import CropioStructuredTool


class CacheTools(BaseModel):
    """Default tools to hit the cache."""

    name: str = "Hit Cache"
    cache_handler: CacheHandler = Field(
        description="Cache Handler for the cropio",
        default_factory=CacheHandler,
    )

    def tool(self):
        return CropioStructuredTool.from_function(
            func=self.hit_cache,
            name=self.name,
            description="Reads directly from the cache",
        )

    def hit_cache(self, key):
        split = key.split("tool:")
        tool = split[1].split("|input:")[0].strip()
        tool_input = split[1].split("|input:")[1].strip()
        return self.cache_handler.read(tool, tool_input)
