"""Command-line module entry point for ``python -m twopy.napari``.

Inputs: command-line arguments parsed by ``twopy.napari.launcher``.
Outputs: a napari app session.
"""

from twopy.napari.launcher import main

if __name__ == "__main__":
    main()
