# Input data templates — for refreshing the reefer model with new data

These templates define the **exact format** the model expects for new reefer
data. Fill a template with a new period's data (e.g. the next month/quarter),
drop the files in, and run the input-refresh — the **model is not retrained**,
it just gets updated inputs to predict each reefer's next trip.

See `docs/INPUT_REFRESH.md` for the full workflow.

## Monthly workflow (the simple version)
1. Fill the matching template below with the **new month's** data.
2. Drop the file into `data/raw/monthly_updates/<type>/` — or upload it on the
   app's **Data Refresh** page.
3. Click **"Update inputs from new data (no retrain)"** in the app (or run
   `python pipeline/refresh_current_state.py`).
4. Open a reefer — its stats and next-trip risk now reflect the new data. The
   trained model is unchanged.

| Template | Drop into | Feeds |
|---|---|---|
| `reefer_unit_repairs_TEMPLATE.csv`     | `monthly_updates/reefer_repairs/`   | repair history, PM compliance |
| `meter_readings_TEMPLATE.csv`          | `monthly_updates/meter_readings/`   | engine hours |
| `container_repair_detail_TEMPLATE.csv`     | `monthly_updates/container_repairs/`| container repair cost |
| `telematics_alarm_history_TEMPLATE.csv`    | `monthly_updates/telematics/`       | shutdown/alarm/low-fuel recency |

## The input files

| Template | Feeds | Canonical source it extends | Format match |
|---|---|---|---|
| `reefer_unit_repairs_TEMPLATE.csv` | repair history, PM compliance | `data/raw/repairs/Updated Reefer Unit Repairs 2024-2025.xlsx` (sheet `Sheet1`) | ✅ identical columns — append new rows |
| `meter_readings_TEMPLATE.csv` | engine hours | `data/raw/hours meter reading/Meter Reading History.xlsx` | ✅ identical — append new rows |
| `container_repair_detail_TEMPLATE.csv` | container repair history | `data/raw/repairs/Container Repair Detail.xlsx` | ✅ identical — append new rows |
| `telematics_alarm_history_TEMPLATE.csv` | shutdown / alarm / low-fuel recency | TK "Alarm History Summary" export | ✅ matches the TK summary layout (4 preamble rows, header on row 5) |

> **Telematics note.** The monthly refresh reads the TK **"Alarm History
> Summary"** export — an *aggregated* file (one row per Vehicle × Alarm Type
> with a count `#` and First/Last Logged), **not** the per-event quarterly
> feed. The template above mirrors that layout: 4 preamble rows, then the
> header on row 5. The loader (`refresh_current_state.py`) does `skiprows=4`
> and matches on `Vehicle Name`, `Alarm Type`, `Severity`, `#`, `First Logged`,
> `Last Logged`; severity `R`=Shutdown, `Y`=Check, `G`=Log. Keep those column
> names and the preamble rows intact. `Vehicle Name` is the container id
> (e.g. `HRTU673864`); its last 6 digits map to the reefer unit.

### Also required for a full RETRAIN (not the monthly refresh)

The monthly refresh above (no retrain) covers repairs, hours, container cost,
and telematics recency. A **full model rebuild** additionally derives **trips**
from the rail-mileage manifest and **shutdown labels** from the per-event
telematics feed. Those two are *not* templatized — provide them in the **same
report layout** as the 2025 baselines:

| Input | Canonical baseline (copy its layout) |
|---|---|
| Rail mileage manifest (one row per leg) | `data/raw/mileage/Rail Miles 2024 and 2025.xlsx`, sheet `Rail Miles Data 2024 & 2025`, header on row 5 |
| Telematics per-event feed (for shutdown labels) | `data/raw/telematics/quarterly/Daily Telematics Update -YTD 2025 Q*.xlsx` |
| Fuel | `data/raw/fuel data/MMA 2025 Fuel Data.xlsx`, sheet `Data` |

> The Term-2 (Jan–Apr 2026) files we received are in **different report
> layouts** (e.g. the "Trailer Mileage Report" for rail miles). To feed those
> through, re-export them in the layouts above, or add a format adapter
> (see `docs/INPUT_REFRESH.md` → "Adapting other report formats").

## Already have the Term-2 (Jan–Apr 2026) drop? It's pre-processed
You don't need to hand-fill these for the Term-2 data we already received.
`pipeline/build_term2_inputs.py` converts `data/raw/TERM 2 datasets/`
into these exact template formats under **`data/processed/term2/`**, with a
binding report. Run it with `--deploy` to stage the CSVs into
`monthly_updates/` for the refresh. See `data/processed/term2/README.md`.

## Conventions
- Reefer unit ids: `R####` (e.g. `R8448`). Container ids: e.g. `HRTU673864`.
- **ID binding:** repairs + meter bind on the reefer id `R####` (must be a unit
  the model tracks); container repairs + mileage + telematics carry a container
  id and bind to a reefer via `models/risk_engine/trailer_map.json`. An id not
  found there keeps its last-known value silently — check the Term-2
  `binding_report.md` for unbound ids.
- Dates: keep the source column's existing format.
- Do **not** rename columns — the loaders match on the header names shown in
  each template.
- New rows **extend** history; you don't re-send old rows (duplicates are
  de-duplicated on ingest).
