---
pretty_name: PolicyEngine UK Microcosm staging telemetry
license: other
---

# PolicyEngine UK Microcosm staging telemetry

This access-controlled dataset repository contains build telemetry only. Access
requests are reviewed individually by PolicyEngine maintainers.

Permitted content is limited to version 2 discovery documents, run manifests,
progress documents, independently versioned event records, calibration progress,
and reviewed aggregate JSON diagnostics or build metadata.

The repository must not contain population H5 files, NumPy archives, source
survey tables, row-level extracts, unrestricted local paths, credentials, or
environment data. Production population releases are stored and published
separately.

Run directories are retained for 90 days while keeping at least the latest ten
runs. A Hugging Face administrator performs and records the monthly cleanup.
