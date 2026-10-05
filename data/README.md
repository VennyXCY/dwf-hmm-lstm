# Data

This repository releases **code only**. The monitoring data are **not
redistributed here**.

## Where the real data are

The raw data and the full cleaning procedure are openly available in the
project's HydroShare resource:

- **HydroShare dataset (DOI):** https://doi.org/10.4211/hs.c9be88f8d04e40df8aeea88dd6d7274f


## What phase 1 reads

Phase 1 reads a single pre-processed sensor file:

```
data/outlet_phase1.csv
```

columns: `Datetime, Temp, NH3, Turb, Cond, pH, Flow`.

This file is the sensor record after the **first, manual** quality-assurance
steps of the cleaning procedure (performed on the raw data, requiring
case-by-case judgement, and therefore not scripted). The remaining, 
automated cleaning steps (values exceeding either the rolling-IQR range 
or the site-specific hard caps were flagged and replaced by linear interpolation.) 
are performed by `script/phase1_baseline/build_baseline_archive.py`.

This exact intermediate file is **not redistributed** in this repository. The
cleaning applied here is a study-specific implementation (rolling-IQR outlier
removal, site-specific hard caps, and linear interpolation) whose parameters and
step order slightly differ from the cleaning procedure of the published HydroShare
dataset. It is an adjustment made by the authors on the raw data to make it more
suitable for this study, and it is deliberately not re-published here in order to
preserve the consistency of the openly archived dataset.

Importantly, this does not compromise reproducibility. The **cleaned dataset
openly available on HydroShare** can be used to reproduce the analysis, after
renaming its columns to the schema above (`Temp, NH3, Turb, Cond, pH, Flow`).
Because that dataset is already cleaned, the automated cleaning step in
`build_baseline_archive.py` is redundant for it and does not materially alter it
(set `SKIP_CLEANING = True` to bypass it). The resulting states and conclusions
are **consistent with those reported here** (minor numerical differences are
expected). The synthetic sample below lets you run the code in the meantime.

## What phase 2 reads

Phase 2 reads **cleaned** sensor files, one per date range, referenced by the
`data_csv` entries in `script/phase2_adaptive/config.py` (the `CYCLES` table),
e.g. `data/OUTLET4-<range>-CLEANED-m3h.csv`.

These are the **same monitoring dataset** as above (from HydroShare), already
cleaned with the same method as phase 1; phase 2 only applies forward/backward 
fill for residual gaps. They are **not redistributed here** for the same 
data-governance reason. Obtain the cleaned record from the HydroShare dataset, 
slice it to the cycle date ranges, and keep the files in `data/`.

## Reusing the cleaning method on other data

The manual pre-processing steps described above are ad hoc, site-specific quality
assurance; they are not the transferable part of the workflow and are not meant
to be reproduced. The cleaning method that is general and reusable begins with
this code: rolling-IQR outlier removal, site-specific hard caps, and linear
interpolation. To apply the framework to a different dataset, treat
`data/outlet_phase1.csv` as the natural entry point of the pipeline: provide
your own sensor record in the schema above at this location (i.e. it plays the
role of the raw input for new data) and run the pipeline, adjusting
`HARD_THRESHOLDS` and other parameters to your site.

## Running the code without the real data (functionality check)

To verify that the pipeline runs, generate a small **synthetic** sample with the
same schema:

```bash
python data/make_synthetic_sample.py          # writes data/synthetic_sensordata.csv
```

Then either set `INPUT_CSV` in `build_baseline_archive.py` to
`data/synthetic_sensordata.csv`, or copy that file to `data/outlet_phase1.csv`.

The synthetic data are entirely artificial and **do not reproduce the paper's
results**; they only exercise the code. This sample mainly exercises phase 1.
To exercise phase 2 with it, point the `CYCLES` dates in the phase-2 config at
the synthetic file's date range (or extend `make_synthetic_sample.py` to cover
your windows); phase 2 also needs the baseline archive produced by phase 1.
