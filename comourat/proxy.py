"""Proxy URL validation shared by the HTTP clients and the page fetcher.

A malformed proxy URL makes every request fail immediately. When a pipeline
drops items on fetch failure, that looks like "no content anywhere". The
validation turns the mistake into an error at construction time.
"""

from __future__ import annotations

from urllib.parse import urlsplit

# Schemes requests can use as an HTTP proxy.
SUPPORTED_PROXY_SCHEMES = {"http", "https", "socks5", "socks5h", "socks4"}


class ProxyConfigurationError(ValueError):
    """The configured proxy URL cannot be used by requests."""


def validate_proxy_url(proxy: str) -> str:
    """Return the proxy URL when usable, otherwise raise.

    Raises `ProxyConfigurationError` for an empty URL, an unsupported
    scheme, or a URL without a host.
    """
    if not proxy or not proxy.strip():
        raise ProxyConfigurationError("proxy URL is empty")
    parts = urlsplit(proxy.strip())
    if parts.scheme not in SUPPORTED_PROXY_SCHEMES:
        raise ProxyConfigurationError(
            f"proxy URL scheme {parts.scheme!r} is not supported; "
            f"expected one of {sorted(SUPPORTED_PROXY_SCHEMES)}"
        )
    if not parts.hostname:
        raise ProxyConfigurationError("proxy URL has no host")
    return proxy.strip()
