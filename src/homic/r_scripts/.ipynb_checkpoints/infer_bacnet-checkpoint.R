library(pheatmap)
library(tidyverse)

meta_data <- read.csv("/Users/matga374/Desktop/saturation_glm_analysis/EEspectrum_metadata.csv")
meta_data2 <- read.csv("/Users/matga374/Desktop/saturation_glm_analysis/Scoring_of_scanned_slides_EED.csv")
meta_data2 <- meta_data2[-(37:39),-1]
meta_data2$Image <- gsub("^01","",meta_data2$Image)
meta_data2$Image <- gsub(". Vsi$","",meta_data2$Image)
meta_data2$Image <- gsub(". vsi $","",meta_data2$Image)
meta_data2$Image[6] <- paste0("K", meta_data2$Image[6])
meta_data2$Image[7] <- paste0("K", meta_data2$Image[7])
meta_data2$Image[8] <- paste0("K", meta_data2$Image[8])
meta_data2 <- meta_data2[-which(meta_data2$Score == "No image detected"),]

meta_data2$id <- substr(meta_data2$Image, 1, 5)                # first 3 characters
meta_data2$img  <- substr(meta_data2$Image, 6, 6) # rest of string

meta_data2$Score <- as.numeric(meta_data2$Score)

meta_data3 <- aggregate(Score ~ id, data = meta_data2, FUN = mean)


all_samps = c("KP003","KP004","KP005","KP008","KP010","KP011","KP012",
             "KP013","KP016","KP021","KP024","KP025","KP026","KP027","KP029",
             "KP033","KP035","KP037","KP038","KP040","KP041","KP046","KP047","KP048","KP049","KP052") 

meta_data <- meta_data[match(all_samps, meta_data$id),]

samp_df <- data.frame(id = all_samps)
meta_data4 <- merge(samp_df, meta_data3, by = "id", all.x = TRUE)


extract_species <- function(file) {
  df <- read.table(file, sep = "\t", quote = "", comment.char = "", fill = TRUE, stringsAsFactors = FALSE)
  df <- df %>%
    filter(V5 < 1e-200, V3 > 99) %>%    
    group_by(V1) %>%
    arrange(V5, desc(V3), .by_group = TRUE) %>%  
    slice(1)
  
  last_col <- df[[ncol(df)]]
  
  cleaned <- last_col |>
    gsub("MAG:\\s*", "", x = _) |>
    
    # Remove brackets ONLY when they contain a single capitalized taxon word,
    # but keep the word inside.
    gsub("\\[([A-Z][a-z]+)\\]", "\\1", x = _) |>
    
    trimws()
  
  # Remove any record containing "sp."
  cleaned[ grepl("\\bsp\\.\\b", cleaned, ignore.case = TRUE) ] <- NA
  
  # Extract first two words
  species <- sub("^([A-Za-z0-9_-]+\\s+[A-Za-z0-9_-]+).*", "\\1", cleaned)
  
  # Drop NA (sp. removed)
  species <- species[!is.na(species)]
  
  return(species)
}




spec_abu_tabs <- list()
df_all <- data.frame()
main_path <- "/Users/matga374/Desktop/saturation_glm_analysis/blastn_refseq/"

files_nams <- list.files(main_path)

for(i in 1:length(files_nams)){

  tmp_spec <- extract_species(paste0(main_path,"/",files_nams[i]))
  tmp_spec_tab <- tmp_spec %>% table
  tmp_spec_tab <- tmp_spec_tab/sum(tmp_spec_tab)
  spec_abu_tabs[[i]] <- tmp_spec_tab
  print(i)
}

all_names <- Reduce(union, lapply(spec_abu_tabs, names))

tbl_list_full <- lapply(spec_abu_tabs, function(tb) {
  x <- tb[all_names]
  x[is.na(x)] <- 0
  x
})

df <- as.data.frame(tbl_list_full)
rownames(df) <- all_names
df

df_abu <- df[ , sapply(df, is.numeric) ]
colnames(df_abu) <- all_samps

# select top 50
top_05perc <- which(sort(rowSums(df_abu)/26, decreasing = T) > 0.005)

df_abu_top <- df_abu[match(names(top_05perc),rownames(df_abu)),]
df_abu_top <- as.matrix(df_abu_top)

sex <- as.factor(meta_data$female)
levels(sex) <- c("male","female")
sex <- as.character(sex)

ses <- as.factor(meta_data$highSES)
levels(ses) <- c("low","high")
ses <- as.character(ses)

sample_info <- data.frame(
  Sex = sex,
  SES = ses
)

rownames(sample_info) <- colnames(df_abu_top)


pheatmap(
  df_abu_top,
  scale = "column",
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  annotation_col = sample_info,
  show_rownames = TRUE,
  show_colnames = TRUE,
  fontsize = 10
)

library(gbm)
library(MASS)


# Build the Boosted Regression Model
set.seed(1)
df_abu_top_t <- t(df_abu_top)
df_lst <- list()

for (i in 1:length(colnames(df_abu_top_t))) {
  
  response <- colnames(df_abu_top_t)[i]            # column to predict
  predictors <- setdiff(colnames(df_abu_top_t), response)
  
  # Build formula: response ~ all other columns
  formula <- as.formula(
    paste0("`", response, "` ~ ", paste0("`", predictors, "`", collapse = " + "))
  )

 df_abu_top_t <- as.data.frame(df_abu_top_t)
 boost_bac <- gbm(formula, distribution = "gaussian", 
                    data=df_abu_top_t, n.trees = 500, interaction.depth=2,
                    shrinkage = 0.1, n.minobsinnode = 2) # learning rate
 var_imp <- summary(boost_bac, plotit = FALSE)
 var_imp$var <- gsub("`","",var_imp$var)

 var_imp_uns <- var_imp[match(colnames(df_abu_top_t)[colnames(df_abu_top_t) != "V1"], var_imp$var), ]
 df_lst[[i]] <- var_imp_uns$rel.inf

}

df_all_imp <- do.call(rbind, df_lst) %>% as.data.frame()
rownames(df_all_imp) <- colnames(df_abu_top_t)
colnames(df_all_imp) <- colnames(df_abu_top_t)

threshold <- 15  # keep only edges with importance > 0.3
adj_pruned <- df_all_imp
adj_pruned[adj_pruned < threshold] <- 0
adj_pruned[is.na(adj_pruned)] <- 0

library(igraph)
adj_pruned <- as.matrix(adj_pruned)


abbreviate_name <- function(x) {
  sapply(strsplit(x, " "), function(words) {
    if(length(words) == 1) return(words)
    paste(words[1], paste(words[-1], collapse = " "), sep = "\n")
  })
}


rownames(adj_pruned) <- abbreviate_name(rownames(adj_pruned))
colnames(adj_pruned) <- abbreviate_name(rownames(adj_pruned))



library(tidygraph)
library(ggraph)

# Build graph
net <- as_tbl_graph(adj_pruned, directed = FALSE) %>% 
  activate(edges) %>% 
  filter(weight > 0) %>% 
  activate(nodes) %>% 
  filter(centrality_degree() > 0)

# Plot
ggraph(net, layout = "fr") +
  geom_edge_link(aes(width = weight), color = "darkgray", alpha = 0.8) +
  scale_edge_width(range = c(0.5, 3), name = "Variable importance") +
  geom_node_point(aes(size = centrality_degree()), color = "#1E90FF") +
  scale_size_continuous(name = "Node degree", range = c(3,10)) +
  geom_node_text(aes(label = name), repel = TRUE, size = 3) +
  theme_void()