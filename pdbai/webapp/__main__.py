"""The web interface's development server, run with python -m pdbai.webapp.

The server is configured from the environment. PDBAI_HOST is the address to
bind, 127.0.0.1 by default, and PDBAI_PORT the port, 8001 by default.
PDBAI_DEBUG turns on Flask's debug mode when it is true, and
PDBAI_TRUSTED_HOSTS lists, by commas, more host names the app answers to.

A specific address other than loopback is added to the trusted host names. A
wildcard address adds none. The app then answers only to loopback names until
PDBAI_TRUSTED_HOSTS lists the others, and the server logs a warning while it
lists none. The port and the debug flag are checked before the app is built.
A bad value therefore stops the server before any workspace directory is
created, as a bad option stops the command line.
"""

from __future__ import annotations

import logging
import os
import sys

from ..console_logging import configure_console_logging
from ..integers import parse_positive_int
from .factory import LOOPBACK_HOSTS, create_app
from .options import parse_flag

# The default is loopback, so nothing else on the network reaches the server
# unless the deployment opts in by setting PDBAI_HOST.
_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 8001
_MAX_PORT = 65535
# Binding one of these listens on every interface, so it names no host a client
# would send.
_WILDCARD_HOSTS = frozenset({"0.0.0.0", "::"})

# Named in full because this module runs as __main__, and console logging
# prints only the pdbai package's loggers.
_logger = logging.getLogger("pdbai.webapp")


def _parse_port(raw_port: str) -> int:
    # A blank value takes the default, and anything that is not a whole number
    # from 1 to _MAX_PORT raises ValueError naming PDBAI_PORT.
    if not raw_port:
        return _DEFAULT_PORT
    port = parse_positive_int(raw_port, "PDBAI_PORT")
    if port > _MAX_PORT:
        raise ValueError(f"PDBAI_PORT must be at most {_MAX_PORT}; got {raw_port!r}.")
    return port


def main() -> int:
    """Serve the web interface until interrupted, and return the exit status.

    A PDBAI_PORT that is not a whole number from 1 to 65535, or a PDBAI_DEBUG
    that is neither true nor false, prints one error line to standard error
    and returns 2 without building the app.
    """
    configure_console_logging()
    host = os.environ.get("PDBAI_HOST", "").strip() or _DEFAULT_HOST
    try:
        port = _parse_port(os.environ.get("PDBAI_PORT", "").strip())
        debug = parse_flag(os.environ.get("PDBAI_DEBUG"), "PDBAI_DEBUG")
    except ValueError as exception:
        print(f"error: {exception}", file=sys.stderr)
        return 2
    extra_trusted_hosts = [
        name.strip()
        for name in os.environ.get("PDBAI_TRUSTED_HOSTS", "").split(",")
        if name.strip()
    ]
    # Serving another interface also means answering to its name.
    if host in _WILDCARD_HOSTS:
        if not extra_trusted_hosts:
            _logger.warning(
                "PDBAI_HOST %s listens on every interface, but the app answers "
                "only to loopback names until PDBAI_TRUSTED_HOSTS lists the others.",
                host,
            )
    elif host not in LOOPBACK_HOSTS:
        extra_trusted_hosts.append(host)
    application = create_app({"TRUSTED_HOSTS": [*LOOPBACK_HOSTS, *extra_trusted_hosts]})
    # The reloader restarts the process on every source edit, which would kill
    # the builds running on its threads and leave their runs stale.
    application.run(host=host, port=port, debug=debug, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
