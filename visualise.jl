using DataFrames, CSV, PlotlyJS

seedrange = 1:10
iterations = 5000
imagepath = "images/"
datapath = "data/"
filename = "infinite_gmm"
infrefs = ["SMC", "NP-MH", "NP-MH-P"]

# Plot from stored data

# get stored data and return as a dictionary
function get_data(infrefs)
    infdata = Dict()
    for infref in infrefs
        infdata[infref] = DataFrame(CSV.File(join([datapath, filename, "-", infref, ".csv"])))
    end
    return infdata
end

# plot all data of infref
function plot_all_data()

    experiments = Dict()

    # get infrefs data from files
    alldata = get_data(infrefs)
    # store infrefs data into experiments
    for infref in infrefs
        df = alldata[infref]
        infval = []
        for seedname in names(df)
            append!(infval, df[!, seedname])
        end
        experiments[infref] = infval
    end

    # plot graph
    p = plot(
        map((infval, infref) -> histogram(x=infval, name=infref), values(experiments), keys(experiments)),
        Layout(
            # autosize=false,
            # size=(100,20)
            # width=100, height=20,
            title=join(["Sampling results for Infinite GMM"]),
            xaxis_title="Number of mixtures",
            yaxis_title="Count",
            marker_size=50,
            xaxis=attr(
                tickvals=1:15)
        )
    )
    savefig(p, join([imagepath, filename, "-all.html"]))
    savefig(p, join([imagepath, filename, "-all.pdf"]))
end

plot_all_data()
