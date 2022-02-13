.PHONY: help clean distclean
.DEFAULT_GOAL := help

define PRINT_HELP_JLSCRIPT
rx = r"^([a-z0-9A-Z_-]+):.*?##[ ]+(.*)$$"
for line in eachline()
    m = match(rx, line)
    if !isnothing(m)
        target, help = m.captures
        println("$$(rpad(target, 20)) $$help")
    end
end
endef
export PRINT_HELP_JLSCRIPT


help:  ## show this help
	@julia -e "$$PRINT_HELP_JLSCRIPT" < $(MAKEFILE_LIST)

clean: ## Clean up build/doc/testing artifacts
	julia -e 'include("clean.jl"); clean()'

distclean: clean ## Restore to a clean checkout state
	julia -e 'include("clean.jl"); distclean()'

test:
	echo "No tests available for GRAPELinesearchAnalysis"
