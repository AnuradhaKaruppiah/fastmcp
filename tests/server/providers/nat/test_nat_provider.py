import os

from fastmcp import FastMCP
from fastmcp.server.providers import NeMoAgentToolkitProvider

CONFIG_PATH = os.environ["NVIDIA_NAT_CONFIG_FILE"]

provider = NeMoAgentToolkitProvider(
    config_path=CONFIG_PATH,
    tool_names=None,  # or ["my_tool", "group_name"]
    namespace=None,   # or "nat"
)

mcp = FastMCP("nat-poc", providers=[provider])

if __name__ == "__main__":
    mcp.run(transport="streamable-http", port=9903)
