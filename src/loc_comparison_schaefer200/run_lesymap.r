library(LESYMAP)
library(ANTsR)
library(RcppCNPy)

# your data and results paths here
lesydata <- "data"
lesyresults <- "results"

lesion_fps_train <- readLines(file.path(lesydata, "simulated", "lesion_fps.txt"))
lesion_fps_test <- readLines(file.path(lesydata, "simulated", "lesion_fps_test.txt"))

# Check if all files in lesion_fps_train exist
missing_files <- lesion_fps_train[!file.exists(lesion_fps_train)]
if (length(missing_files) > 0) {
    stop("The following files do not exist:\n", paste(missing_files, collapse = "\n"))
}

behavior_train <- npyLoad(file.path(lesydata, "simulated_schaefer200", "Y.npy"))
behavior_test <- npyLoad(file.path(lesydata, "simulated_schaefer200", "Y_test.npy"))

template <- antsImageRead(Sys.glob(file.path(lesydata, "vol_mask_2mm.nii.gz"))[1])

for (i in seq_len(ncol(behavior_train))) {
    print(sprintf("Running lesymap for behavior %d", i))
    try({
        savedir <- file.path(lesyresults, sprintf("simulated_schaefer200/lesymap_results/run_%d", i))
        if (!dir.exists(savedir)) {
            dir.create(savedir, recursive = TRUE)
        }
        lsm <- lesymap(lesion_fps_train,
                       behavior_train[, i],
                       method = "sccan",
                       optimiseSparseness = TRUE,
                       saveDir = savedir)

    }, silent = FALSE)
}