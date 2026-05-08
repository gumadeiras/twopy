"""Shared user-facing error text helpers for napari widgets.

Inputs: exceptions caught at GUI boundaries.
Outputs: concise diagnostic text suitable for dialogs and status messages.

Napari callbacks should show enough exception detail for users to diagnose bad
paths or malformed recordings without duplicating formatting in each widget.
"""


def exception_message_for_user(error: Exception) -> str:
    """Return exception class and message for user-visible diagnostics.

    Args:
        error: Exception caught at a user-facing GUI boundary.

    Returns:
        Text containing the exception class and message, or only the class name
        when the exception has no message.
    """
    message = str(error)
    if message == "":
        return type(error).__name__
    return f"{type(error).__name__}: {message}"
