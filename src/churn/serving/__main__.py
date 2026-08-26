"""Local entry point: ``python -m churn.serving``.

Exists so that the host and port configured for the deployment are the ones the
server actually binds, instead of being declared in one place and passed on the
command line in another.

``uvicorn`` is imported inside :func:`main` rather than at module level. The
serving environment has it; the root analysis environment does not, and importing
this package must not require it.
"""

from __future__ import annotations

import logging

from churn.serving.settings import load_settings


def main() -> None:
    """Start the server on the configured host and port.

    Raises:
        ServingStartupError: Propagated from the application's startup gates. The
            process exits rather than serving an unverified model.
    """
    import uvicorn

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    uvicorn.run(
        "churn.serving.api:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":  # pragma: no cover - process entry point
    main()
