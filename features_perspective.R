#######################################################
# This file computes toxicity scores and further features using the Perspective API
# https://perspectiveapi.com/
#######################################################

## Set up environment
# Clean environment
rm(list = ls())

# Load packages
library(peRspective)
library(tidyverse)
library(foreach)
library(doParallel)

# Set seed
set.seed(42)

# Setup parallel backend
cores <- detectCores()
cl <- makeCluster(cores-1)

# Define paths to data
path_input <- "../../../Data/input/parleys_user/"
path_output <- "../../../Data/output/features_perspective/"

# Function to truncate posts to long for API
truncate_posts <- function(string){
  for (i in 1:length(string)) {
    while (nchar(string[i], type = "bytes") >= 20480) {
      string[i] <- str_trunc(string[i], width = (nchar(string[i], type = "chars") - 1), side = "right", ellipsis = "")
    }
  }
  return(string)
}

# List files to analyze with Perspctive API
file_list_perspective <- list.files(path_input, pattern = "csv")

# Files already processed
file_list_processed <- list.files(str_c(path_output, "raw/"), pattern = "csv")

# Files to process
file_list_unprocessed <- str_c("user_content_", setdiff(parse_number(file_list_perspective), parse_number(file_list_processed)), ".csv")

registerDoParallel(cl)

# Retrieve perspctive scores
foreach(i=1:length(file_list_unprocessed), .combine = "c", .packages = c("tidyverse", "peRspective")) %dopar% {
  
  # Load data
  dta <- read_csv(str_c(path_input, file_list_unprocessed[i])) %>%
    distinct() %>%
    filter(typeof(body) == "character" & !is.na(body)) %>%
    mutate(body = ifelse(nchar(body, type = "bytes") >= 20480, truncate_posts(body), body))
  
  # Extract toxicity scores from Perspective API
  text_sample <- dta %>%
    prsp_stream(text = body,
                text_id = id,
                score_model = c("TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"),
                safe_output = FALSE,
                languages = "en",
                sleep = 0.00009
                  ) %>%
      left_join(subset(dta, select = c(creator, id)), by = c("text_id" = "id"))
    
    # Save raw data
    write_csv(text_sample, file = str_c(path_output, "raw/features_perspective_", parse_number(file_list_unprocessed[i]), ".csv"))
    
    dta_agg <- text_sample %>%
      group_by(creator) %>%
      summarize(across(.cols = c("TOXICITY", "SEVERE_TOXICITY", "IDENTITY_ATTACK", "INSULT", "PROFANITY", "THREAT"), ~ mean(.x, na.rm = TRUE)))
    
    # Save features by user file
    write_csv(dta_agg, file = str_c(path_output, "by_user/features_perspective_", parse_number(file_list_unprocessed[i]), ".csv"))
  
  
  print(str_c("Processed file ", i, " of ", length(file_list_perspective)))
  
}
stopCluster(cl)

# Concatenate files and create final feature set
features_liwc <- list.files(path = str_c(path_output, "by_user/"), pattern = "csv") %>%
  str_c(path_output, "by_user/", .) %>%
  map_dfr(~ read_csv(.x))

write_csv(features_liwc, file = "../../../Data/output/features_perspective.csv")


