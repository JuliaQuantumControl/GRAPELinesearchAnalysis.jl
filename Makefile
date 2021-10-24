.PHONY: help clean distclean
.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
    match = re.match(r'^([a-z0-9A-Z_-]+):.*?## (.*)$$', line)
    if match:
        target, help = match.groups()
        print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

help:  ## show this help
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

clean: ## Clean up build/doc/testing artifacts
	rm -f src/*.cov

distclean: clean ## Restore to a clean checkout state
	rm -f Manifest.toml

test:
	echo "No tests available for GRAPELinesearchAnalysis"
