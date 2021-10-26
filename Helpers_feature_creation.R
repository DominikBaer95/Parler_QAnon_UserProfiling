######################################
## Helper functions text embeddings
######################################

# Clean text data and save cleaned file for embeddings
clean_text_embedding <- function(file_name, path_input, path_output){
  # Load data
  text <- read_rds(file = str_c(path_input, file_name)) %>%
    select(creator, body, id)
  
  # Clean parleys and split by user
  text_clean <- text %>%
    mutate(body = str_replace_all(body, "@[a-z,A-Z,0-9]*", ""),
           body = str_replace_all(body, "[^\x01-\x7F]", ""),
           body = str_replace_all(body, "[^[:alnum:][:space:].]", ""),
           body = str_replace_all(body, "\\s+", " ")) %>%
    filter(body != "" & body != " ") %>%
    group_by(creator)
  
  # Save files
  write_csv(text_clean, str_c(path_output, str_remove(file_name, pattern = ".RDS"), "_clean_embeddings", ".csv"))
  
  return(str_c("Saved file ", file_name))
} 

# Compute embeddings by user
compute_embeddings <- function(file_name, path_input, path_output){
  # Load data
  text_clean <- read_rds(file = str_c(path_input, file_name))
  
  embeddings <- text_clean %>%
    future_map(~ textEmbed(.x$body,
                           model = "bert-base-uncased",
                           layers = 12,
                           contexts = TRUE,
                           context_aggregation_layers = "mean",
                           context_aggregation_tokens = "mean",
                           decontexts = FALSE)$x, furrr_options(seed = TRUE))
  
  embeddings_final <- embeddings %>%
    bind_rows() %>%
    bind_cols(subset(bind_rows(text_clean), select = creator))
  
  saveRDS(embeddings_final, str_c(path_output, "embeddings_", parse_number(file_name), ".RDS"))
  
  return(str_c("Saved file ", file_name))
}

# Aggregate embeddings by user
aggregate_embeddings <- function(file_name, path_input, path_output){
  embeddings_raw <- read_csv(file = str_c(path_input, file_name))
  
  embeddings_user <- embeddings_raw %>%
    select(-1) %>%
    rename_with(~ str_c("text", c(0:383)), .cols = c(-creator)) %>%
    group_by(creator) %>%
    summarize(across(starts_with("text"), ~ mean(.x)))
  
  saveRDS(embeddings_user, str_c(path_output, "embeddings_aggregated_user", parse_number(file_name), ".RDS"))
  
  return(str_c("Saved file ", file_name))
}

######################################
## Helper function annotation
######################################

annotate_text <- function(file_name, path_input, path_output){
  text <- read_rds(file = str_c(path_input, file_name)) %>%
    select(creator, body, id) %>%
    mutate(comment_id = row_number())
  
  annotation <- udpipe_annotate(udmodel_en, text[["body"]]) %>%
    as.data.frame() %>%
    select(doc_id, token, upos) %>%
    mutate(doc_id = as.numeric(str_remove(doc_id, pattern = "doc")))
  
  tidy_parleys_annotated <- annotation %>%
    mutate(token = str_to_lower(token, locale = "en"),
           token_lag = lag(token),
           # Handle split of "wwg1wga"
           token = if_else(str_detect(token, pattern = "wwg") & str_detect(lead(token), pattern = "1") & str_detect(lead(token, n = 2), pattern = "wga"), "wwg1wga", token, missing = token),
           upos = if_else(token == "1" & lag(token) == "wwg1wga" & lead(token) == "wga", "DROP", upos, missing = token),
           upos = if_else(token == "wga" & lag(token, n = 2) == "wwg1wga" & lag(token) == "1", "DROP", upos, missing = token),
           token = if_else(token == "wwg1" & lead(token) == "wga", "wwg1wga", token, missing = token),
           upos = if_else(token == "wga" & lag(token) == "wwg1wga", "DROP", upos, missing = token),
           token = if_else(str_detect(token_lag, pattern = "^@$|^#$"), str_c(token_lag, token, sep = ""), token, missing = token), # merge single @/# with next line 
           upos = if_else(str_detect(token, pattern = "@|#"), "X", upos), # label # -> X, @ -> X
           upos = if_else(str_detect(token, pattern = "^@$|^@@$|^#$|^##$"), "DROP", upos),
           token = if_else(str_detect(token, pattern = "&") & upos == "CCONJ", "and", token), # "&" to "and"
           token = str_replace_all(token, pattern = "[^\x01-\x7F]", "")) %>% # Remove Emojis
    select(-token_lag) %>%
    left_join(text, by = c("doc_id" = "comment_id")) %>% # merge creator
    rename(comment_id = doc_id) %>%
    select(-c(body, id)) %>%
    filter(!str_detect(token, pattern = "^@$|^#$") 
           & !str_detect(token, pattern = "^'s$")
           & upos != "PUNCT" 
           & upos != "DROP" 
           & upos != "SYM"
           & upos != "NUM") %>% # Filter punctuation, single @/#, Symbols/Emojis, Numbers
    group_by(comment_id, creator) %>%
    mutate(token = str_replace_all(token, c("^n't$|^n't$|^nt$" = "not", 
                                            "^'m$" = "am",
                                            "^'re$" = "are",
                                            "^'ll$" = "will",
                                            "^'ve$" = "have")),
           token = str_remove_all(token, pattern = "\\.|\\!|\\?")) %>%
    filter(token != "")
  
  saveRDS(tidy_parleys_annotated, file = str_c(path_output, "annotated_content_", parse_number(file_name), ".RDS"))
  
  return(str_c("Saved file ", file_name))
}

########################################
## Helper functions linguistic features
########################################

create_linguistic_meta_content <- function(file_name_tidy, file_name_content, path_input, path_output = NULL){
  
  # Load tidy parleys
  tidy_parleys_annotated <- read_rds(file = str_c(path_input, "output/annotation/", file_name_tidy)) %>%
    group_by(comment_id, creator)
  
  # Load conent data
  data_qanon_full <- read_rds(file = str_c(path_input, "input/parleys_user/", file_name_content)) %>%
    group_by(comment_id, creator)
  
  # Vector of important POS-tags
  pos_tags <- c("NOUN", "PROPN", "PRON", "ADJ", "ADV", "VERB", "AUX", "ADP", "DET")
  
  content <- tidy_parleys_annotated %>%
    mutate(word_count = row_number(), # counter for words in each document
           hash = if_else(str_detect(token, "#"), 1, 0), # Indicator hashtag
           handle = if_else(str_detect(token, "@"), 1, 0), # Indicator handle
           rel_pos = if_else(upos %in% pos_tags, 1, 0), # Indicator for relevant pos tags
           long_words = if_else(str_count(token) > 6 & upos != "X", 1, 0)) %>% # Indicator of long words
    select(-token)
  
  # Content features
  features_parleys <- content %>%
    group_by(comment_id, creator) %>%
    summarize(n_handle = sum(handle),
              n_hash = sum(hash),
              n_pos = sum(rel_pos),
              prop_pos = sum(rel_pos)/max(word_count),
              n_long_words = sum(long_words),
              prop_long_words = sum(long_words)/max(word_count))
  
  # Content features
  features_parleys_user <- features_parleys %>%
    group_by(creator) %>%
    summarize(#n_handle_user = sum(n_handle),
      #n_hash_user = sum(n_hash),
      #n_pos_user = sum(n_pos),
      #n_long_words_user = sum(n_long_words),
      freq_handle_user = sum(n_handle)/n(),
      freq_hash_user = sum(n_hash)/n(),
      freq_pos_user = sum(n_pos)/n(),
      freq_long_words_user = sum(n_long_words)/n())
  
  # Features from posts (datatype == post)
  ## Number of upvotes and impressions per post
  features_post_user <- data_qanon_full %>%
    filter(datatype == "posts") %>%
    select(comment_id, creator, upvotes, impressions) %>%
    group_by(creator) %>%
    summarize(n_upvotes_posts = sum(upvotes),
              n_impressions = sum(impressions),
              freq_upvotes_posts = sum(upvotes)/n(),
              freq_impressions = sum(impressions)/n(),
              n_posts = n())
  
  # Features from comments (datatype == comments)
  ## Number of upvotes, downvotes, and Average score (upvotes - downvotes) per comment
  features_comments_user <- data_qanon_full %>%
    filter(datatype == "comments") %>%
    select(comment_id, creator, upvotes, downvotes, score) %>%
    group_by(creator) %>%
    mutate(downvotes = if_else(str_detect(downvotes, pattern = "k") == TRUE, as.numeric(str_remove(downvotes, pattern = "k"))*1000, as.numeric(downvotes))) %>% # Format downvotes if like 1.2k
    summarize(n_upvotes_comments = sum(upvotes),
              n_downvotes_comments = sum(downvotes),
              n_score = sum(score),
              freq_upvotes_comments = sum(upvotes)/n(),
              freq_downvotes_comments = sum(downvotes)/n(),
              freq_score = sum(score)/n(),
              n_comments = n())
  
  # URLs
  features_url_user <- data_qanon_full %>%
    select(comment_id, creator, bodywithurls) %>%
    mutate(urls = if_else(str_detect(bodywithurls, pattern = "(http|https)[^([:blank:]|\\'|<|&|#\n\r)]+"), 1, 0)) %>%
    group_by(creator) %>%
    summarize(n_urls = sum(urls),
              freq_urls = sum(urls)/n())
  
  # Sentiment
  parleys_raw <- get_sentences(data_qanon_full %>% pull(body))
  features_sentiment <- sentiment_by(parleys_raw, polarity_dt = lexicon::hash_sentiment_nrc) %>%
    select(element_id, sentiment_score = ave_sentiment) %>%
    cbind(data_qanon_full) %>%
    select(creator, sentiment_score) %>%
    mutate(comment_id = row_number())
  
  features_sentiment_user <- content %>%
    group_by(comment_id) %>%
    summarize(n_words = max(word_count, na.rm = TRUE)) %>%
    inner_join(features_sentiment, by = c("comment_id")) %>%
    group_by(creator) %>%
    summarise(user_sentiment = sum((n_words/sum(n_words)) * sentiment_score))
  
  list_stylistic_features <- list(features_parleys_user, features_post_user, features_comments_user, features_url_user, features_sentiment_user)
  
  # Create final 
  features_content_user <- list_stylistic_features %>%
    reduce(full_join, by = c("creator")) %>%
    mutate(set = parse_number(file_name_tidy))
  
  saveRDS(features_content_user, file = str_c(path_output, "features_content_", parse_number(file_name_tidy), ".RDS"))
  
  return(str_c("Saved file ", file_name_tidy))
  #return(features_content_user)
}


