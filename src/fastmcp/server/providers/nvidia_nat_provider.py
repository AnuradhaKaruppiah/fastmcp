"""Provider for NVIDIA NeMo Agent Toolkit workflows."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from inspect import Parameter
from pathlib import Path
from typing import Any

from fastmcp.server.providers.local_provider import LocalProvider
from fastmcp.tools.function_tool import FunctionTool
from fastmcp.tools.tool import Tool
from fastmcp.utilities.logging import get_logger
from fastmcp.utilities.versions import VersionSpec

logger = get_logger(__name__)


def _require_nvidia_nat() -> None:
    """Raise a clear error when the NVIDIA NeMo Agent Toolkit dependency is missing."""
    try:
        import nat  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "NeMo Agent Toolkit is required for NeMoAgentToolkitProvider. "
            "Install the optional dependency with `fastmcp[nvidia-nat]`."
        ) from exc


class NeMoAgentToolkitProvider(LocalProvider):
    """Provider that exposes NeMo Agent Toolkit workflows as FastMCP tools."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        tool_names: Sequence[str] | None = None,
        namespace: str | None = None,
    ) -> None:
        super().__init__(on_duplicate="replace")
        self._config_path = Path(config_path)
        self._tool_names = list(tool_names) if tool_names else None
        self._namespace = namespace
        self._loaded = False
        self._load_lock: asyncio.Lock | None = None
        self._exit_stack: AsyncExitStack | None = None
        self._session_managers: dict[str, Any] = {}

        if namespace:
            from fastmcp.server.transforms import Namespace

            self.add_transform(Namespace(namespace))

    async def _ensure_loaded(self) -> None:
        if self._loaded:
            return

        if self._load_lock is None:
            self._load_lock = asyncio.Lock()

        async with self._load_lock:
            if not self._loaded:
                await self._load_tools()
                self._loaded = True

    async def _load_tools(self) -> None:
        _require_nvidia_nat()

        from nat.builder.workflow import Workflow
        from nat.builder.workflow_builder import WorkflowBuilder
        from nat.runtime.loader import load_config
        from nat.runtime.session import SessionManager
        from nat.plugins.mcp.server.tool_converter import (
            create_function_wrapper,
            get_function_description,
        )

        config = load_config(self._config_path)
        self._exit_stack = AsyncExitStack()
        builder = await self._exit_stack.enter_async_context(
            WorkflowBuilder.from_config(config=config)
        )

        workflow = await builder.build()
        functions = await self._get_all_functions(workflow)
        functions = self._filter_functions(functions)

        session_managers: dict[str, SessionManager] = {}
        for function_name, function in functions.items():
            if isinstance(function, Workflow):
                session_managers[function_name] = await SessionManager.create(
                    config=config,
                    shared_builder=builder,
                    entry_function=None,
                )
            else:
                session_managers[function_name] = await SessionManager.create(
                    config=config,
                    shared_builder=builder,
                    entry_function=function_name,
                )

        for function_name, session_manager in session_managers.items():
            target_function = functions.get(function_name)
            input_schema = getattr(
                target_function, "input_schema", session_manager.workflow.input_schema
            )
            wrapper = create_function_wrapper(
                function_name, session_manager, input_schema
            )
            self._ensure_wrapper_annotations(wrapper)
            description = get_function_description(target_function or session_manager.workflow)
            tool = FunctionTool.from_function(
                wrapper,
                name=function_name,
                description=description,
            )
            self.add_tool(tool)

        self._session_managers = session_managers

    async def _list_tools(self) -> Sequence[Tool]:
        await self._ensure_loaded()
        return await super()._list_tools()

    async def _get_tool(
        self, name: str, version: VersionSpec | None = None
    ) -> Tool | None:
        await self._ensure_loaded()
        return await super()._get_tool(name, version)

    async def _get_all_functions(self, workflow: Any) -> dict[str, Any]:
        functions: dict[str, Any] = {}
        functions.update(workflow.functions)
        for function_group in workflow.function_groups.values():
            functions.update(await function_group.get_accessible_functions())

        if workflow.config.workflow.workflow_alias:
            functions[workflow.config.workflow.workflow_alias] = workflow
        else:
            functions[workflow.config.workflow.type] = workflow

        return functions

    def _filter_functions(self, functions: Mapping[str, Any]) -> dict[str, Any]:
        if not self._tool_names:
            return dict(functions)

        filtered_functions: dict[str, Any] = {}
        for function_name, function in functions.items():
            if function_name in self._tool_names:
                filtered_functions[function_name] = function
            elif any(
                function_name.startswith(f"{group_name}.")
                for group_name in self._tool_names
            ):
                filtered_functions[function_name] = function
        return filtered_functions

    def _ensure_wrapper_annotations(self, wrapper: Any) -> None:
        signature = getattr(wrapper, "__signature__", None)
        if signature is None:
            return

        annotations: dict[str, Any] = dict(getattr(wrapper, "__annotations__", {}))
        for name, parameter in signature.parameters.items():
            if parameter.annotation is Parameter.empty:
                continue
            annotations.setdefault(name, parameter.annotation)

        if signature.return_annotation is not Parameter.empty:
            annotations.setdefault("return", signature.return_annotation)

        wrapper.__annotations__ = annotations

    async def _shutdown(self) -> None:
        for session_manager in self._session_managers.values():
            try:
                await session_manager.shutdown()
            except Exception:
                logger.exception("Failed to shut down NeMo Agent Toolkit session manager")
        self._session_managers = {}

        if self._exit_stack is not None:
            await self._exit_stack.aclose()
            self._exit_stack = None

        self._loaded = False

    @asynccontextmanager
    async def lifespan(self):
        await self._ensure_loaded()
        try:
            yield
        finally:
            await self._shutdown()
