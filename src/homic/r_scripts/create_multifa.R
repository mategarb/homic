library(biomartr)
library(seqinr)

options(timeout = 30000)

############################################################
# USER SETTINGS
############################################################

input_file <- "/file.txt"
output_dir <- "/folder"

full_genome <- TRUE
seq_lens <- c(1000)   # used only if full_genome == FALSE

file_name <- "3prime_1000_genome"

############################################################
# READ SPECIES LIST
############################################################

specs <- read.table(input_file, sep = ",", stringsAsFactors = FALSE)
specs <- unname(unlist(as.vector(specs)))

if (full_genome) {
  seq_lens <- 0
}

############################################################
# HELPER FUNCTION
############################################################

download_genome <- function(db_name, assembly_accession) {
  
  gnm <- NULL
  
  for (k in 1:5) {
    try({
      gnm <- getGenome(
        db = db_name,
        organism = assembly_accession,
        path = file.path("_ncbi_downloads", "genomes"),
        reference = FALSE,
        mute_citation = TRUE
      )
      message("Genome retrieved")
      break
    }, silent = TRUE)
    
    Sys.sleep(5)
  }
  
  return(gnm)
}

############################################################
# MAIN LOOP
############################################################

for (ii in 1:length(seq_lens)) {
  
  seq_len <- seq_lens[ii]
  
  dna_seq_vec <- c()
  source_vec <- c()
  strain_vec <- c()
  date_vec <- c()
  srcdb_vec <- c()
  assembly_level <- c()
  full_name <- c()
  taxid_vec <- c()
  
  for (i in 1:length(specs)) {
    
    dna_seq <- NA
    
    ########################################################
    # REFSEQ FIRST
    ########################################################
    
    check1 <- is.genome.available(
      db = "refseq",
      organism = specs[i],
      details = TRUE,
      skip_bacteria = FALSE
    )
    
    if (length(dim(check1)) == 2) {
      
      db_used <- "refseq"
      
      # Priority 1: reference genome
      if (length(which(check1$refseq_category == "reference genome")) != 0) {
        
        check1 <- check1[which(check1$refseq_category == "reference genome"), ]
        source_label <- "reference genome"
        
        # Priority 2: complete genome
      } else if (length(which(check1$assembly_level == "Complete Genome")) != 0) {
        
        check1 <- check1[which(check1$assembly_level == "Complete Genome"), ]
        source_label <- "complete genome"
        
        # Priority 3: newest assembly
      } else {
        
        newest_idx <- which.max(as.Date(check1$seq_rel_date))
        check1 <- check1[newest_idx, ]
        source_label <- "newest assembly"
      }
      
      gnm <- download_genome(
        db_name = "refseq",
        assembly_accession = check1$assembly_accession[1]
      )
      
      if (!is.null(gnm)) {
        
        gnm_seq <- read_genome(gnm)
        gnm_seq <- as.data.frame(gnm_seq)
        
        tmp_name <- rownames(gnm_seq)
        gnm_seq <- as.character(gnm_seq[nrow(gnm_seq), 1])
        
        if (full_genome) {
          dna_seq <- gnm_seq
        } else {
          if (nchar(gnm_seq) >= seq_len) {
            # 3-prime end extraction
            dna_seq <- substr(
              gnm_seq,
              nchar(gnm_seq) - seq_len + 1,
              nchar(gnm_seq)
            )
          }
        }
        
        source_vec[i] <- source_label
        full_name[i] <- tmp_name
        strain_vec[i] <- check1$infraspecific_name[1]
        date_vec[i] <- check1$seq_rel_date[1]
        srcdb_vec[i] <- db_used
        assembly_level[i] <- check1$assembly_level[1]
        taxid_vec[i] <- check1$taxid[1]
      }
      
    } else {
      
      ######################################################
      # FALLBACK TO GENBANK
      ######################################################
      
      check2 <- is.genome.available(
        db = "genbank",
        organism = specs[i],
        details = TRUE,
        skip_bacteria = FALSE
      )
      
      if (length(dim(check2)) == 2) {
        
        db_used <- "genbank"
        
        # Priority 1
        if (length(which(check2$refseq_category == "reference genome")) != 0) {
          
          check2 <- check2[which(check2$refseq_category == "reference genome"), ]
          source_label <- "reference genome"
          
          # Priority 2
        } else if (length(which(check2$assembly_level == "Complete Genome")) != 0) {
          
          check2 <- check2[which(check2$assembly_level == "Complete Genome"), ]
          source_label <- "complete genome"
          
          # Priority 3
        } else {
          
          newest_idx <- which.max(as.Date(check2$seq_rel_date))
          check2 <- check2[newest_idx, ]
          source_label <- "newest assembly"
        }
        
        gnm <- download_genome(
          db_name = "genbank",
          assembly_accession = check2$assembly_accession[1]
        )
        
        if (!is.null(gnm)) {
          
          gnm_seq <- read_genome(gnm)
          gnm_seq <- as.data.frame(gnm_seq)
          
          tmp_name <- rownames(gnm_seq)
          gnm_seq <- as.character(gnm_seq[nrow(gnm_seq), 1])
          
          if (full_genome) {
            dna_seq <- gnm_seq
          } else {
            if (nchar(gnm_seq) >= seq_len) {
              # 3-prime end extraction
              dna_seq <- substr(
                gnm_seq,
                nchar(gnm_seq) - seq_len + 1,
                nchar(gnm_seq)
              )
            }
          }
          
          source_vec[i] <- source_label
          full_name[i] <- tmp_name
          strain_vec[i] <- check2$infraspecific_name[1]
          date_vec[i] <- check2$seq_rel_date[1]
          srcdb_vec[i] <- db_used
          assembly_level[i] <- check2$assembly_level[1]
          taxid_vec[i] <- check2$taxid[1]
        }
      }
    }
    
    if (is.na(dna_seq)) {
      dna_seq_vec[i] <- NA
      source_vec[i] <- NA
      strain_vec[i] <- NA
      date_vec[i] <- NA
      srcdb_vec[i] <- NA
      assembly_level[i] <- NA
      full_name[i] <- NA
      taxid_vec[i] <- NA
    } else {
      dna_seq_vec[i] <- dna_seq
    }
    
    print(paste0(
      "processed ", i,
      " out of ", length(specs), " genomes"
    ))
  }
  
  ##########################################################
  # REMOVE FAILURES
  ##########################################################
  
  nmis <- which(
    is.na(dna_seq_vec) |
      is.na(full_name)
  )
  
  if (length(nmis) == 0) {
    dna_seq_lst <- as.list(dna_seq_vec)
    seq_names <- full_name
    taxids <- taxid_vec
    assembly_levels_clean <- assembly_level
  } else {
    dna_seq_lst <- as.list(dna_seq_vec[-nmis])
    seq_names <- full_name[-nmis]
    taxids <- taxid_vec[-nmis]
    assembly_levels_clean <- assembly_level[-nmis]
  }
  
  ##########################################################
  # KRAKEN HEADERS
  ##########################################################
  
  kraken_headers <- sapply(seq_along(seq_names), function(i) {
    
    h <- seq_names[i]
    t <- taxids[i]
    
    parts <- strsplit(h, " ", fixed = TRUE)[[1]]
    id <- parts[1]
    desc <- if (length(parts) > 1)
      paste(parts[-1], collapse = " ")
    else ""
    
    paste0(
      id,
      "|kraken:taxid|",
      t,
      if (desc != "") paste0(" ", desc) else ""
    )
    
  }, USE.NAMES = FALSE)
  
  ##########################################################
  # SAVE FILES
  ##########################################################
  
  fname <- paste0(
    output_dir,
    file_name,
    "_",
    seq_len,
    ".fasta"
  )
  
  write.fasta(
    dna_seq_lst,
    kraken_headers,
    fname
  )
  
  fname2 <- paste0(
    output_dir,
    file_name,
    "_",
    seq_len,
    "_comGenOnly.fasta"
  )
  
  ncg <- which(assembly_levels_clean == "Complete Genome")
  
  if (length(ncg) > 0) {
    write.fasta(
      dna_seq_lst[ncg],
      kraken_headers[ncg],
      fname2
    )
  }
}
