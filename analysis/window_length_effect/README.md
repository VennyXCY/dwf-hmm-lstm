# Effect of rolling window length

Sensitivity analysis comparing a 3-day and a 5-day rolling window configuration
across the update cycles.

## Script

`plot_window_length_effect.py`  (env: any)
Reads two performance histories and draws the 3-day vs 5-day comparison across
four indicators (mean matching confidence, mean predictive entropy,
95th-pct KLD, state switching rate), with reconstruction-triggered cycles marked.
→ `results/window_length_effect.{png,pdf}`

## Inputs

Two full Phase 2 runs, one per window configuration:

1. Run the Phase 2 pipeline with the **3-day** window configuration (W=9, S=3,
   H=3 in `CYCLES`), then copy `results/performance_history.csv` to
   `results/performance_history_3day.csv`.
2. Run it again with the **5-day** configuration (e.g. W=15, S=5, H=5), then copy
   `results/performance_history.csv` to `results/performance_history_5day.csv`.

Both histories must carry the enriched columns (`Conf_Mean_Raw`, `H_Mean`,
`KLD_95`, `Switches_Per_Hour`, `Reconstruction_Triggered`), which the pipeline's
`diagnostics.py` now writes.

## Notes

- Heaviest analysis: the whole Phase 2 pipeline must be run twice over all
  cycles, once with the 3-day `CYCLES` and once with a 5-day `CYCLES`
  (W=15, S=5, H=5). The window length is set by the dates in `CYCLES`, not by a
  numeric parameter.
