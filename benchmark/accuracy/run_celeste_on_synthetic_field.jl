#!/usr/bin/env julia

import ArgParse
using DataFrames

import Celeste.AccuracyBenchmark
import Celeste.Infer
import Celeste.ParallelRun
import Celeste.SDSSIO

const OUTPUT_DIRECTORY = joinpath(splitdir(Base.source_path())[1], "output")

arg_parse_settings = ArgParse.ArgParseSettings()
ArgParse.@add_arg_table arg_parse_settings begin
    "--fft"
        help = "Use FFT inference"
        action = :store_true
    "--limit-num-sources"
        help = "Target only the given number of sources, for quicker testing"
        arg_type = Int
    "--image-fits"
        help = "FITS file containing synthetic imagery; if not specified, will use default SDSS RCF"
    "catalog_csv"
        help = "CSV catalog for initialization (as generated by write_ground_truth_catalog.csv)"
        required = true
end
parsed_args = ArgParse.parse_args(ARGS, arg_parse_settings)

srand(12345)

if parsed_args["image-fits"] != nothing
    extensions = AccuracyBenchmark.read_fits(parsed_args["image_fits"])
    images = AccuracyBenchmark.make_images(extensions)
else
    images = SDSSIO.load_field_images(
        [AccuracyBenchmark.STRIPE82_RCF],
        AccuracyBenchmark.SDSS_DATA_DIR,
    )
end
@assert length(images) == 5

catalog_data = AccuracyBenchmark.read_catalog(parsed_args["catalog_csv"])
catalog_entries = AccuracyBenchmark.make_initialization_catalog(catalog_data)
@printf("Loaded %d sources...\n", length(catalog_entries))

if parsed_args["limit-num-sources"] != nothing
    target_sources = collect(1:parsed_args["limit-num-sources"])
else
    target_sources = collect(1:length(catalog_entries))
end
neighbor_map = Infer.find_neighbors(target_sources, catalog_entries, images)
results = ParallelRun.one_node_joint_infer(
    catalog_entries,
    target_sources,
    neighbor_map,
    images,
    use_fft=parsed_args["fft"],
)

results_df = AccuracyBenchmark.celeste_to_df(results)

if parsed_args["image-fits"] != nothing
    catalog_label = splitext(basename(parsed_args["image_fits"]))[1]
else
    rcf = AccuracyBenchmark.STRIPE82_RCF
    catalog_label = @sprintf("sdss_%s_%s_%s", rcf.run, rcf.camcol, rcf.field)
end
output_filename = joinpath(OUTPUT_DIRECTORY, @sprintf("%s_predictions.csv", catalog_label))
@printf("Writing results to %s\n", output_filename)
AccuracyBenchmark.write_catalog(output_filename, results_df)
