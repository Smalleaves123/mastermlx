from __future__ import annotations

import numpy as np

from mastermlx.data import DataContract
from mastermlx.tabular import DataReadinessReport


def main():
    train = np.array(
        [
            [22.0, 0.1],
            [35.0, 0.4],
            [48.0, 0.8],
            [61.0, 0.9],
        ]
    )
    incoming = np.array(
        [
            [25.0, 0.2],
            [130.0, 0.6],
            [42.0, np.nan],
            [42.0, np.nan],
        ]
    )
    contract = DataContract(
        rules={
            "x0": {"kind": "numeric", "min": 0.0, "max": 120.0},
            "x1": {"kind": "numeric", "min": 0.0, "max": 1.0, "missing_rate": 0.25},
        }
    )
    report = DataReadinessReport(data_contract=contract).fit(train).run(incoming)

    print("status:", report.status)
    print("ready:", report.ready)
    print("issues:", report.issues)
    print("missing:", report.quality["missing"]["count"])


if __name__ == "__main__":
    main()
