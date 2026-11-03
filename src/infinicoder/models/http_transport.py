"""HTTP requests retain the caller's endpoint authorization."""

import urllib.request


class NoRedirect(urllib.request.HTTPRedirectHandler):
    """A redirect does not authorize sending credentials or source to a new URL."""

    def redirect_request(self, *args, **kwargs):
        return None


def open_request(request, *, timeout):
    """Preserve normal proxy/TLS handling while refusing automatic redirects."""
    return urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout)
