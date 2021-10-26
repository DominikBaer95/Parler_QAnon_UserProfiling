# Split data into files, each containing all observations for 1000 users

#######################################################################################

library(tidyverse)
library(tidytext)
library(textclean)
library(furrr)

#######################################################################################
# Set Seed
set.seed(1234)

# Input data
path_input <- "./../data/input/"
# Output data
path_output <- "./../data/output/"

content_vars <- c("id", "comments", "body", "bodywithurls", "createdAtformatted", "creator", "datatype", "hashtags", "media", "upvotes", "impressions", "reposts", "downvotes", "score", "urls")

# Helper function
# Read specific columns of Rdata files and join with dataset
read_rdata_join <- function(filename, join_dta, key){
  load(filename) 
  out <- dta %>%
    inner_join(join_dta, by = key)
  return(out)
}

# Data content
file_list_content <- list.files(path =  str_c(path_input, "parler_data/Rdata/"), pattern = "*.Rdata")

# Load QAnon supporters
load(str_c(path_output, "user_qanon.Rdata"))


# Number of users per sample
splits <- 1000

# List with different users
user_splits <- user_qanon %>%
  select(creator) %>%
  split((seq(nrow(user_qanon))-1) %/% splits) %>%
  flatten()

# Name user_content files
names_user_content <- str_c("user_content_", 1:length(user_splits))
names(user_content) <- str_c("user_content_", 1:length(user_content))

# Save raw user_content files
names(user_content) %>%
  map2(.y = user_content, ~ saveRDS(.y, file = str_c(path_input, "parleys_user/", .x, ".RDS")))

# Function to attach new parleys to exisitng dataframe
attach_data <- function(names_user_content, user_splits, input_data){
  dta <- read_rds(str_c(path_input, "parleys_user/", names_user_content, ".RDS"))
  
  new_parleys <- input_data %>%
    filter(creator %in% user_splits)
  
  dta <- bind_rows(dta, new_parleys)
  
  saveRDS(dta, file = str_c(path_input, "parleys_user/", names_user_content, ".RDS"))
  
  print(str_c("Finished dataset ", names_user_content))
  return(NULL)
}

# Assign content of each user to specific dataset
for (i in 1:length(file_list_content)) {
  # Load parleys
  content <- read_rdata_join(filename = str_c(path_input, "parler_data/Rdata/", file_list_content[i]), join_dta = user_qanon, key = "creator")
  
  print(str_c("Loaded data ", i, " of ", length(file_list_content)))
  
  # Tidy parleys
  parleys <- content %>%
     filter(body != "") %>%
     mutate(body = replace_contraction(body, contraction.key = lexicon::key_contractions)
     ) 
  
  print(str_c("Cleaned data ", i, " of ", length(file_list_content), " at ", Sys.time()))
  
  future_map2(.x = names_user_content, .y = user_splits, ~ attach_data(.x, .y, input_data = parleys))
}
