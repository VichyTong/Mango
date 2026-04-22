from typing import Callable
from .simple_navigation import web_navigation as simple_web_navigation
from .mcp_navigation import web_navigation as mcp_web_navigation


async def get_navigator(navigator_type: str = "simple") -> Callable:
    """
    Get navigator function based on type.

    Args:
        navigator_type: Type of navigator ("simple" or "mcp")

    Returns:
        Navigator function (web_navigation)

    Raises:
        ValueError: If navigator_type is not recognized
    """
    if navigator_type == "simple":
        return simple_web_navigation
    elif navigator_type == "mcp":
        return mcp_web_navigation
    else:
        raise ValueError(f"Unknown navigator type: {navigator_type}. Choose 'simple' or 'mcp'")
