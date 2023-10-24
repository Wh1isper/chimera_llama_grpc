import functools
import typing

import anyio.to_thread
import importlib_metadata

T = typing.TypeVar("T")


async def run_in_threadpool(
    func: typing.Callable[..., T], *args: typing.Any, **kwargs: typing.Any
) -> T:
    """
    This function is copied from starlette.concurrency.run_in_threadpool
    BSD 3-Clause License
    https://github.com/encode/starlette/blob/0.31.1/starlette/concurrency.py#29
    """
    if kwargs:  # pragma: no cover
        # run_sync doesn't accept 'kwargs', so bind them in here
        func = functools.partial(func, **kwargs)
    return await anyio.to_thread.run_sync(func, *args)


def load_entry_point(distribution, group, name):
    dist_obj = importlib_metadata.distribution(distribution)
    eps = [ep for ep in dist_obj.entry_points if ep.group == group and ep.name == name]
    if not eps:
        raise ImportError("Entry point %r not found" % ((group, name),))
    return eps[0].load()
