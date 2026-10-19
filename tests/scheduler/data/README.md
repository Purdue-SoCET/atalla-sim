# Scheduler test fixtures

## `atalla_isa_sheet.csv`

Snapshot of the Atalla ISA spreadsheet — the authoritative instruction set:

https://docs.google.com/spreadsheets/d/1yDJ_oH0EXGIE4-4wVcwTeaw1Bg1vpoUSIkgTK3qDw_w/edit?gid=1072084504

Fetched 2026-09-25 as CSV (`/export?format=csv&gid=1072084504`). Snapshotted
rather than fetched at test time so the suite runs offline and a change to the
sheet shows up as a diff here, not as a test that fails for no local reason.

To refresh:

```bash
curl -sSL -o tests/scheduler/data/atalla_isa_sheet.csv "https://docs.google.com/spreadsheets/d/1yDJ_oH0EXGIE4-4wVcwTeaw1Bg1vpoUSIkgTK3qDw_w/export?format=csv&gid=1072084504"
```

`test_isa.py::test_the_table_matches_the_spec_sheet` then reports any opcode
the model and the spec disagree on.
