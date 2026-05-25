from typing import Any
from .registry import ToolRegistry
from ..schemas.common import ToolSchema


def get_order_status(order_id: str) -> dict[str, Any]:
    return {"order_id": order_id, "status": "shipped", "eta": "2 days"}


def cancel_order(order_id: str, reason: str = "user_request") -> dict[str, Any]:
    return {"order_id": order_id, "cancelled": True, "reason": reason}


def update_address(order_id: str, new_address: str) -> dict[str, Any]:
    return {"order_id": order_id, "address_updated": True, "new_address": new_address}


_SCHEMAS: list[ToolSchema] = [
    ToolSchema(
        name="get_order_status",
        description="Get the current status of an order by order ID.",
        required_fields=["order_id"],
        properties={"order_id": {"type": "string"}},
    ),
    ToolSchema(
        name="cancel_order",
        description="Cancel an order. Requires explicit user confirmation.",
        required_fields=["order_id"],
        optional_fields=["reason"],
        properties={
            "order_id": {"type": "string"},
            "reason": {"type": "string"},
        },
    ),
    ToolSchema(
        name="update_address",
        description="Update the shipping address for an order.",
        required_fields=["order_id", "new_address"],
        properties={
            "order_id": {"type": "string"},
            "new_address": {"type": "string"},
        },
    ),
]

_HANDLERS = {
    "get_order_status": get_order_status,
    "cancel_order": cancel_order,
    "update_address": update_address,
}


def build_default_registry() -> ToolRegistry:
    registry = ToolRegistry()
    for schema in _SCHEMAS:
        registry.register(schema, _HANDLERS[schema.name])
    return registry


DEFAULT_REGISTRY: ToolRegistry = build_default_registry()
