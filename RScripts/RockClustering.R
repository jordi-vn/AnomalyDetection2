#!/usr/bin/env Rscript
setwd("/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
#step1 <- new.env(parent = baseenv())
list.of.packages <- c("proxy", "cba")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages(lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")[,"Package"])]
if(length(new.packages)) install.packages(new.packages, lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
args = commandArgs(trailingOnly=TRUE)
library(utils)
library(proxy, lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
library(cba, lib = "/home/jverdu/SB_NLP/AnomalyDetection/Rpackages")
#library(dummies)
#library(devtools)

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
  df[sapply(df, is.character)] <- lapply(df[sapply(df, is.character)], 
                                         as.factor)
  df_bin <- as.dummy(df)
  result.rock <-rockCluster(df_bin, as.numeric(args[2]), theta = as.numeric(args[3]))
  #Save clustering result into csv
  write.csv(result.rock$cl, file = args[4])
}
