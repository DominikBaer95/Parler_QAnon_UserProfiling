###################################################################
# This file computes linguistic features using the LIWC dictionary
# https://www.liwc.app/
###################################################################

## Set up environment
# Clean environment
rm(list = ls())

# Load packages
library(quanteda)
library(quanteda.dictionaries)
library(tidyverse)

# Set seed
set.seed(42)

# Define paths to data
path_input <- "../../../Data/input/parleys_user/"
path_output <- "../../../Data/output/features_liwc/"

# Load LIWC dictionary
load("LIWC2015/liwc2015_dict.RData")

# List files to analyze with LIWC
file_list_liwc <- list.files(path_input, pattern = "RDS")

# Extract LIWC features
for (i in 1:length(file_list_liwc)) {
  
  # Load data
  dta <- read_rds(str_c(path_input, file_list_liwc[i])) %>%
    select(creator, body, id) %>%
    distinct() %>%
    filter(typeof(body) == "character") 
  
  # Number of parleys
  n_parleys <- dta %>%
    group_by(creator) %>%
    summarize(n_parleys = n())
  
  # Preprocessing
  dta <- dta %>%
    mutate(body = str_replace_all(body, "@[a-z,A-Z,0-9]*", ""),
           body = str_replace_all(body, "[^\x01-\x7F]", ""),
           body = str_replace_all(body, "[^[:alnum:][:space:][:punct:]]", ""), # Keep punctuation for LIWC as this is a dimension of the feature space.
           body = str_replace_all(body, "\\s+", " ")) %>%
    filter(body != "" & body != " ") %>%
    rowid_to_column() 
  
  # Extract features
  features_liwc <- dta %>%
    corpus(text_field = "body") %>%
    liwcalike(dictionary = liwc_dict) %>%
    mutate(text_id = parse_number(docname)) %>%
    left_join(subset(dta, select = c("rowid", "creator", "id")), by = c("text_id" = "rowid")) %>%
    select(-c(docname, text_id))
  
  # Save raw data
  write_csv(features_liwc, file = str_c(path_output, "raw/features_liwc_", i, ".csv"))
    
  dta_agg <- features_liwc %>%
    left_join(n_parleys, by = "creator") %>%
    group_by(creator) %>%
    summarize(across(.cols = WPS:OtherP, ~ sum(.x, na.rm = TRUE)/max(n_parleys)))
  
  # Save features by user file
  write_csv(dta_agg, file = str_c(path_output, "by_user/features_liwc_", i, ".csv"))
  
  print(str_c("Processed file ", i, " of ", length(file_list_liwc)))
  
}

# Concatenate files and create final feature set
features_liwc <- list.files(path = str_c(path_output, "by_user/"), pattern = "csv") %>%
  str_c(path_output, "by_user/", .) %>%
  map_dfr(~ read_csv(.x))

write_csv(features_liwc, file = "../../../Data/output/features_liwc.csv")
