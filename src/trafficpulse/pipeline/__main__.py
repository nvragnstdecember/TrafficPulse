"""``python -m trafficpulse.pipeline`` entry point (P1-U12).

Delegates to the offline wrong-way slice runner (the composition root). Importing
``trafficpulse.pipeline`` itself still pulls in no backend -- only executing this
module (or calling :func:`trafficpulse.pipeline.runner.main`) wires the real
RT-DETR detector + IoU tracker + persistence together.
"""

from .runner import main

if __name__ == "__main__":
    raise SystemExit(main())
