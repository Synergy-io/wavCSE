# Offline validation for the public model and worker repository.
UV := uv run --locked

.PHONY: check model-check clustering-check import-check

check: model-check clustering-check import-check

model-check:
	$(UV) python -m unittest discover -s improvements/taskrelation/tests -t .

clustering-check:
	$(UV) python -m unittest discover -s improvements/clustering/tests -t .

import-check:
	$(UV) python -m improvements.run_improvements --help
