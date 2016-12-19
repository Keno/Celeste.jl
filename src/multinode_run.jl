using Base.Threads
using Gasp


"""
Use Dtree to distribute the passed bounding boxes to multiple nodes for
processing. Within each node, process the light sources in each of the
assigned boxes with multiple threads.
"""
function multi_node_infer(boxes::Vector{BoundingBox},
                          stagedir::String;
                          outdir=".",
                          primary_initialization=true,
                          timing=InferTiming())
    nwi = length(boxes)
    each = ceil(Int64, nwi / nnodes)

    if nodeid == 1
        nputs(nodeid, "running on $nnodes nodes")
        nputs(nodeid, "$nwi bounding boxes, ~$each per node")
    end

    # per-thread timing
    ttimes = Array(InferTiming, nthreads())

    # results
    results = OptimizedSource[]
    results_lock = SpinLock()

    # create Dtree and get the initial allocation
    dt, is_parent = Dtree(nwi, 0.4)
    ni, (ci, li) = initwork(dt)
    rundt = runtree(dt)

    # work item processing loop
    nputs(nodeid, "dtree: initial: $ni ($ci to $li)")

    ilock = SpinLock()
    function process_tasks()
        tid = threadid()
        ttimes[tid] = InferTiming()
        times = ttimes[tid]

        if rundt && tid == 1
            ntputs(nodeid, tid, "dtree: running tree")
            while runtree(dt)
                cpu_pause()
            end
        else
            while true
                lock(ilock)
                if li == 0
                    unlock(ilock)
                    ntputs(nodeid, tid, "dtree: out of work")
                    break
                end
                if ci > li
                    ntputs(nodeid, tid, "dtree: consumed allocation (last was $li)")
                    ni, (ci, li) = getwork(dt)
                    unlock(ilock)
                    ntputs(nodeid, tid, "dtree: $ni work items ($ci to $li)")
                    continue
                end
                item = ci
                ci = ci + 1
                unlock(ilock)

                box = boxes[item]

                tic()
                rcfs = get_overlapping_fields(box, stagedir)
                timing.query_fids = timing.query_fids + toq()

                catalog, sources, images, neighbor_map
                    = infer_init(rcfs, stagedir;
                                 objid=objid,
                                 primary_initialization=primary_initialization,
                                 timing=timing)

                try
                    s = sources[ts]
                    entry = catalog[s]
                    Log.debug("processing source $s: objid = $(entry.objid)")

                    # could subset images to images_local here too.
                    neighbors = catalog[neighbor_map[ts]]

                    t0 = time()
                    vs_opt = infer_source_callback(images, neighbors, entry)
                    runtime = time() - t0

                    result = OptimizedSource(entry.thing_id,
                                             entry.objid,
                                             entry.pos[1],
                                             entry.pos[2],
                                             vs_opt)
                    lock(results_lock)
                    push!(results, result)
                    unlock(results_lock)
                catch ex
                    if is_production_run || nthreads() > 1
                        Log.error(string(ex))
                    else
                        rethrow(ex)
                    end
                end
            end
        end

        tic()
        save_results(outdir, box, results)
        itimes.write_results = toq()

        timing.num_infers = timing.num_infers+1
        add_timing!(timing, itimes)
        rundtree(rundt)
    end
    nputs(nodeid, "out of work")
    tic()
    while rundt[]
        rundtree(rundt)
    end
    finalize(dt)
    timing.wait_done = toq()

    tic()
    if nthreads() == 1
        process_sources()
    else
        ccall(:jl_threading_run, Void, (Any,), Core.svec(process_sources))
        ccall(:jl_threading_profile, Void, ())
    end
    timing.opt_srcs = toq()
    timing.num_srcs = length(target_sources)

    return results
end


