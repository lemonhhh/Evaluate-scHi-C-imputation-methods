library(HiCImpute)
library(tidyverse)
suppressPackageStartupMessages(library(ComplexHeatmap))
suppressPackageStartupMessages(library(circlize))
library(reticulate)
options(digits=2)
library(reticulate)
np <- import("numpy")
library(abind)


reconstruct_matrix <- function(vector) {
  n <- (1 + sqrt(1 + 8 * length(vector))) / 2
  n <- as.integer(n)
  mat <- matrix(0, n, n)
  mat[upper.tri(mat, diag = FALSE)] <- vector
  mat[lower.tri(mat)] <- t(mat)[lower.tri(mat)]
  return(mat)
}

extract_number <- function(file_name) {
  matches <- regmatches(file_name, regexec("copy(\\d+)", file_name))
  as.numeric(matches[[1]][2])  # 提取并返回数字部分
}

downsample_matrix_dir <- "/shareb/mliu/evaluate_impute/data/simulation_hic/Cell2020/hic/matrix"
group_list <- c(1:2)


for (group in group_list){
  print(group)
  filename = paste("Cell2020_sample_matrix",group,".npy",sep="")
  sc_data <- np$load(paste(downsample_matrix_dir,"/",filename,sep=""))
  #前100个
  sc_data<-sc_data[1:100, , ]
  ndim <- dim(sc_data)[2]

  sum_data <- apply(sc_data, c(2, 3), sum)
  scHiC <- aperm(sc_data, c(2, 3, 1))
  
  result=MCMCImpute(scHiC=scHiC
                        ,bulk=sum_data
                        ,expected=NULL
                        ,startval=c(100,100,10,8,10,0.1,900,0.2,0,replicate(dim(scHiC)[3],8))
                        ,n=ndim
                        ,mc.cores=50
                        ,cutoff=0.5
                        ,niter=30000
                        ,burnin=5000)
  all_matrices <- list()
  all_matrices <- sapply(1:ncol(result$Impute_All), function(col_index) {
    # v <- result$Impute_SZ[, col_index]
    v <- result$Impute_All[, col_index]
    matrix_result <- reconstruct_matrix(v)
    all_matrices[[col_index]] <- matrix_result
    return(matrix_result)
  }, simplify = FALSE)

  large_matrix <- abind(all_matrices, along = 3)
  large_matrix <- aperm(large_matrix, c(3, 1, 2))
  np$save(paste0("/shareb/mliu/evaluate_impute/data/imputed_data/hicimpute/hicimputed_Cell2020_",group,".npy"), large_matrix)
}