#!/usr/bin/env Rscript
setwd("/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
list.of.packages <- c("dbscan")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages(lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
args = commandArgs(trailingOnly=TRUE)
library(dbscan, lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
#Arguments should be path to OHE dataframe, expected number of clusters and theta parameter

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).n", call.=FALSE)
} else
{
  print(args)
  #Load One-hot-encoded dataframe
  df <- read.csv(args[1], sep = ",", header = TRUE)[,-1]
  # Convert all variables to factors
  #df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], 
  #                                       as.factor)
  #df_ohe <- model.matrix(~.-1, data=df)
  #df_bin <- as.dummy(df)
  result.dbscan <- hdbscan(df, minPts = as.numeric(args[2]))
  #Save clustering result into csv
  write.csv(result.dbscan$cluster, file = args[3])
}
