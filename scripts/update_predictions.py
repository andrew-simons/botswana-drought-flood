"""Monthly update: re-run export_app_data.py after adding a new month to the cube.

Usage (after downloading the new month's GEE data to data/cube/):
    python scripts/update_predictions.py

--- FUTURE AUTOMATION (GitHub Actions) ---
Create .github/workflows/monthly_update.yml with:

    on:
      schedule:
        - cron: '0 6 1 * *'   # 06:00 UTC on the 1st of each month
      workflow_dispatch:        # allow manual trigger

    jobs:
      update:
        runs-on: ubuntu-latest
        steps:
          - uses: actions/checkout@v4
          - uses: actions/setup-python@v5
            with: { python-version: '3.12' }
          - run: pip install -r requirements.txt
          - run: python scripts/gee_export_latest_month.py   # (you write this)
          - run: python scripts/update_predictions.py
          - uses: stefanzweifel/git-auto-commit-action@v5
            with:
              commit_message: "chore: monthly prediction update"
              file_pattern: "data/app/* app/assets/*"

The `gee_export_latest_month.py` script (to be written) should:
  1. Authenticate GEE with a service account key stored in GitHub Secrets
  2. Export the new month's CHIRPS/ERA5/MODIS/SMAP arrays
  3. Append the new month to data/cube/dynamic.npy and data/cube/labels.npy
  4. Update data/cube/meta.json (increment T, update end date)
"""

import subprocess
import sys
from pathlib import Path

# Just re-run the full export — it regenerates all prediction files.
# Safe to call repeatedly; idempotent.
script = Path(__file__).parent / "export_app_data.py"
sys.exit(subprocess.call([sys.executable, str(script)]))
