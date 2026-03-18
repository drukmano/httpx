from __future__ import annotations

import inspect
import typing

from quent import Q

from .._models import Request, Response
from .base import BaseTransport

SyncHandler = typing.Callable[[Request], Response]
AsyncHandler = typing.Callable[[Request], typing.Coroutine[None, None, Response]]


__all__ = ["MockTransport"]


class MockTransport(BaseTransport):
    def __init__(self, handler: SyncHandler | AsyncHandler) -> None:
        self.handler = handler

    def _do_handle_request(
        self, request: Request, *, _async: bool = False
    ) -> typing.Any:
        return (
            Q(request)
            .do(lambda r: r.aread() if _async else r.read())
            .then(lambda r: self.handler(r))
            .run()
        )

    def handle_request(self, request: Request) -> Response:
        response = self._do_handle_request(request)
        if not isinstance(response, Response):  # pragma: no cover
            raise TypeError("Cannot use an async handler in a sync Client")
        return response

    async def handle_async_request(self, request: Request) -> Response:
        result = self._do_handle_request(request, _async=True)
        if inspect.isawaitable(result):
            return await result
        return result
