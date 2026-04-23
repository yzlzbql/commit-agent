from __future__ import annotations

from ..types import TodoItem, ToolResult


def seed(task: str) -> list[TodoItem]:
    return [
        TodoItem(content=f"Understand task: {task}", status="in_progress", priority="high"),
        TodoItem(content="Gather evidence or make the required progress", status="pending", priority="high"),
        TodoItem(content="Check completion and deliver the final result", status="pending", priority="high"),
    ]


def on_tool(items: list[TodoItem], res: ToolResult) -> list[TodoItem]:
    if res.metadata.get("todo_items"):
        return [TodoItem.model_validate(item) for item in res.metadata["todo_items"]]
    if not items:
        return items
    items[0].status = "completed"
    if len(items) > 1 and items[1].status == "pending":
        items[1].status = "in_progress"
    if "verify" in res.title.lower() and len(items) > 2:
        items[2].status = "in_progress"
    return items


def on_final(items: list[TodoItem]) -> list[TodoItem]:
    if len(items) > 1:
        items[1].status = "completed"
    if len(items) > 2:
        items[2].status = "in_progress"
    return items


def on_verify(items: list[TodoItem], ok: bool) -> list[TodoItem]:
    if ok:
        for item in items:
            item.status = "completed"
        return items
    if len(items) > 2:
        items[2].status = "pending"
    return items
