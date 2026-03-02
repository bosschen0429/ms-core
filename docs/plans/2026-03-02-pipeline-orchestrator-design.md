# Pipeline Orchestrator Design

> Date: 2026-03-02

## Overview

Implement `MSPipeline` — a 12-step orchestrator that chains preprocessing,
calibration, processing, and analysis modules. Supports step-by-step execution
with snapshots/undo, or one-shot `run_all()`.

## Architecture

### Core Classes

- `MSPipeline` — stateful orchestrator with snapshot dict
- `PipelineConfig` — dataclass holding all step parameters + `skip_steps`
- `StepDef` — registry entry (number, name, label, source, execute callable, defaults)

### Module Interface Adapters

| Module type | Native API | Pipeline wrapper |
|---|---|---|
| processing/ | `func(df, **kw) → df` | Direct call, wrap with MSDataset |
| preprocessing/ | `Processor.process(df) → ProcessingResult` | Extract `.data` from result |
| calibration/ | `main(file_path) → dict` | tempfile Excel round-trip |

### Step Registry (12 steps)

```
1  data_organize         preprocessing/data_organizer
2  duplicate_remove      preprocessing/duplicate_remover
3  ms_quality_filter     preprocessing/ms_quality_filter
4  missing_value         processing/missing_values
5  istd_correction       preprocessing/istd_marker + calibration/istd_correction
6  qc_lowess             calibration/qc_lowess
7  batch_effect           calibration/batch_effect
8  feature_filter        processing/feature_filter
9  pqn_normalize         calibration/pqn
10 transform             processing/transformation
11 scaling               processing/scaling
12 analysis              (not part of pipeline — called separately)
```

### Key Decisions

- Calibration modules wrapped via tempfile Excel round-trip (no refactor needed)
- Snapshots stored per step for undo
- `skip_steps` in config to skip inapplicable steps
- Step 12 (analysis) not included in pipeline — called separately by GUI
