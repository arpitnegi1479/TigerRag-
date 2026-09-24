# Benchmark Dataset v1.0

This is a deliberately constructed synthetic corpus for evaluating the Agentic GraphRAG backend. It is not real-world information and must not be presented as such.

## Corpus design

- `01_program_brief.txt`: Helios Research Institute, Aster Program, Meridian Labs, Atlas Sensor, Nereid Basin, and Zephyr Station.
- `02_meridian_atlas.txt`: Atlas Sensor development, delivery, Zephyr processing, and 2023/2024 timing.
- `03_zephyr_nereid.txt`: Zephyr monitoring, Nereid Basin, Lumen Shelf, methane, and water-vapor observations.
- `04_celeste_analysis.txt`: Atlas/Zephyr evidence analysis and the causal survey-priority chain.
- `05_license_record.txt`: the claim that Meridian Labs licensed Atlas Sensor to Helios Research Institute.
- `06_counter_record.txt`: the deliberate contradictory claim that Meridian did not license Helios and instead licensed Nova Dynamics.

## Gold graph annotations

The documents are ingested through the real pipeline first. Because the Gemini generation quota may force the low-confidence heuristic extractor, the benchmark setup also writes a small, explicit gold annotation layer for relationships whose typed semantics are required by the question manifest. These annotations use the real graph repository and preserve the exact source document/chunk provenance; they are benchmark labels, not claims that the fallback extractor inferred those relationship types.

The contradiction is therefore represented by two distinct provenance-scoped claims between Meridian Labs and Helios Research Institute: `LICENSES` from document 05 and `NOT_LICENSED_TO` from document 06.
