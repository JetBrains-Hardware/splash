"""Entry point of the frozen server that build_server.sh builds.

It takes the arguments of `splash serve` (install/launcher.py) at the top
level: `splash-server --model <owner/repo or DIR> [options]`. A frozen
program has no interpreter to run Splash's other Python programs with, so
the launcher runs them in this executable, by a first argument naming one
(install.paths.program):

  install-models ...     install/models.py, the model installer
  server ...             server/server.py, against the bundled engine the
                         launcher names with --binary (install.paths.BINARY)
  document-worker ...    server/document_worker.py, the PDF worker

The launcher's options all start with --, so none is mistaken for one.
"""

import sys

PROGRAMS = ("install-models", "server", "document-worker")


def main():
    program = sys.argv[1] if sys.argv[1:2] and sys.argv[1] in PROGRAMS else None
    if program is None:
        from install import launcher

        return launcher.main(["serve", *sys.argv[1:]])
    del sys.argv[1]
    if program == "install-models":
        from install import models

        return models.main()
    if program == "document-worker":
        from server import document_worker

        document_worker.main()
        return 0
    from server import server

    server.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
