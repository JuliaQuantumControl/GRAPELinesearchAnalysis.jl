"""
    clean([distclean=false])

Clean up build/doc/testing artifacts. Restore to clean checkout state
(distclean)
"""
function clean(; distclean=false, _exit=true)

    _glob(folder, ending) =
        [name for name in readdir(folder; join=true) if (name |> endswith(ending))]
    _exists(name) = isfile(name) || isdir(name)
    _push!(lst, name) = _exists(name) && push!(lst, name)

    ROOT = @__DIR__

    ###########################################################################
    CLEAN = String[]
    for folder in ["", "src"]
        append!(CLEAN, _glob(joinpath(ROOT, folder), ".cov"))
    end
    append!(CLEAN, _glob(ROOT, ".info"))
    ###########################################################################

    ###########################################################################
    DISTCLEAN = String[]
    _push!(DISTCLEAN, joinpath(ROOT, "Manifest.toml"))
    ###########################################################################

    for name in CLEAN
        @info "rm $name"
        rm(name, force=true, recursive=true)
    end
    if distclean
        for name in DISTCLEAN
            @info "rm $name"
            rm(name, force=true, recursive=true)
        end
        if _exit
            @info "Exiting"
            exit(0)
        end
    end

end

distclean() = clean(distclean=true)
