library(biomartr)
library(seqinr)

options(timeout = 30000)

specs <- read.table('/Users/matga374/Desktop/016_proj_shm/65sp.txt', sep=",")
specs <- unname(unlist(as.vector(specs)))

dna_seq_vec <- c()
source_vec <- c()
strain_vec <- c()
date_vec <- c()
srcdb_vec <- c()
assembly_level <- c()
full_name <- c()
taxid_vec <- c()

seq_lens <- 0 #c(125, 250, 500, 1000, 1500, 2000, 2500) #
#seq_lens <- c(500, 1000, 2500, 5000) #500
full_genome = T
file_name = "full_genome_mm_65sp"
for(ii in 1:length(seq_lens)){
seq_len <- seq_lens[ii]

for(i in 1:length(specs)){
check1 <- is.genome.available(db = "refseq", organism = specs[i], details = TRUE, skip_bacteria = FALSE)
if(length(dim(check1)) == 2){
  if(length(which(check1$refseq_category=="reference genome")) != 0){ # first take the one marked as reference genome
    
    check1 <- check1[which(check1$refseq_category=="reference genome"),]
    gnm <- getGenome(db = "refseq", 
                     organism = check1[1,]$assembly_accession,  
                     path   = file.path("_ncbi_downloads","genomes"), 
                     reference = FALSE)
    
    gnm_seq <- read_genome(gnm)
    gnm_seq <- as.data.frame(gnm_seq)
    tmp_name <- rownames(gnm_seq)
    gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
    
    if(nchar(gnm_seq) < seq_len){
      dna_seq <- NA
      source_vec[i] <- NA
      full_name[i] <- NA
    }else{
      #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
      if(full_genome){
        dna_seq <- gnm_seq
      }else{
      dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
      }
      
      source_vec[i] <- "reference genome"
      full_name[i] <- tmp_name
    }

    
  }else{ # filter and take the newest one
    
    if(length(which(check1$assembly_level == "Complete Genome")) == 1){
      check1 <- check1[which(check1$assembly_level == "Complete Genome"),]
      gnm <- getGenome(db = "refseq", 
                       organism = check1[1,]$assembly_accession,  
                       path   = file.path("_ncbi_downloads","genomes"), 
                       reference = FALSE)
      

      gnm_seq <- read_genome(gnm)
      gnm_seq <- as.data.frame(gnm_seq)
      tmp_name <- rownames(gnm_seq)
      gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
      
      #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
      dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
      
      #dna_seq <- as.character(gnm_seq[[1]][1:seq_len]) # 5 prime
      source_vec[i] <- "complete genome"
    }else{
      
      # take based on the most recent date
      check1 <- check1[which(check1$seq_rel_date == max(as.Date(check1$seq_rel_date))),]
      gnm <- getGenome(db = "refseq", 
                       organism = check1[1,]$assembly_accession,  
                       path   = file.path("_ncbi_downloads","genomes"), 
                       reference = FALSE)
      
      gnm_seq <- read_genome(gnm)
      gnm_seq <- as.data.frame(gnm_seq)
      tmp_name <- rownames(gnm_seq)
      gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
      
      #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
      if(full_genome){
        dna_seq <- gnm_seq
      }else{
        dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
      }

      
      source_vec[i] <- "newest assembly"
      full_name[i] <- tmp_name
    }

  }

dna_seq_vec[i] <- dna_seq
strain_vec[i] <- check1[1,]$infraspecific_name
date_vec[i] <- check1[1,]$seq_rel_date
srcdb_vec[i] <- "refseq"
assembly_level[i] <- check1[1,]$assembly_level
taxid_vec[i] <- check1[1,]$taxid

}else{
  check2 <- is.genome.available(db = "genbank", organism = specs[i], details = TRUE, skip_bacteria = FALSE)
  if(length(dim(check1)) == 2){
    if(length(which(check2$refseq_category=="reference genome")) != 0){ # first take the one marked as reference genome
      
      check2 <- check2[which(check2$refseq_category=="reference genome"),]
      gnm <- getGenome(db = "refseq", 
                       organism = check1[1,]$assembly_accession,  
                       path   = file.path("_ncbi_downloads","genomes"), 
                       reference = FALSE)
      
      gnm_seq <- read_genome(gnm)
      #gnm_seq <- read_rna(gnm)
      gnm_seq <- as.data.frame(gnm_seq)
      tmp_name <- rownames(gnm_seq)
      gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
      #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
      dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
      
      source_vec[i] <- "reference genome"
      full_name[i] <- tmp_name
    }else{ # filter and take the newest one
      
      if(length(which(check2$assembly_level == "Complete Genome")) == 1){
        check2 <- check2[which(check2$assembly_level == "Complete Genome"),]
        gnm <- getGenome(db = "refseq", 
                         organism = check1[1,]$assembly_accession,  
                         path   = file.path("_ncbi_downloads","genomes"), 
                         reference = FALSE)
        
        #gnm <- getRNA(db = "refseq", 
        #              organism = check1[1,]$assembly_accession,  
        #              path   = file.path("_ncbi_downloads","genomes"), 
        #              reference = FALSE)
        gnm_seq <- read_genome(gnm)
        #gnm_seq <- read_rna(rnm)
        gnm_seq <- as.data.frame(gnm_seq)
        tmp_name <- rownames(gnm_seq)
        gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
        #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
        if(full_genome){
          dna_seq <- gnm_seq
        }else{
          dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
        }
        
        source_vec[i] <- "complete genome"
        full_name[i] <- tmp_name
      }else{
        
        # take based on the most recent date
        check2 <- check2[which(check2$seq_rel_date == max(as.Date(check2$seq_rel_date))),]
        gnm <- getGenome(db = "refseq", 
                         organism = check1[1,]$assembly_accession,  
                         path   = file.path("_ncbi_downloads","genomes"), 
                         reference = FALSE)
        
        #gnm <- getRNA(db = "refseq", 
        #              organism = check1[1,]$assembly_accession,  
        #              path   = file.path("_ncbi_downloads","genomes"), 
        #              reference = FALSE)
        gnm_seq <- read_genome(gnm)
        #gnm_seq <- read_rna(rnm)
        gnm_seq <- as.data.frame(gnm_seq)
        tmp_name <- rownames(gnm_seq)
        gnm_seq <- gnm_seq[dim(gnm_seq)[1],]
        #dna_seq <- as.character(gnm_seq[[1]][(length(gnm_seq[[1]]) - seq_len + 1):(length(gnm_seq[[1]]))]) # 3 prime
        if(full_genome){
          dna_seq <- gnm_seq
        }else{
          dna_seq <- substr(gnm_seq, (nchar(gnm_seq) - seq_len + 1), (nchar(gnm_seq))) # 3 prime
        }
        
        source_vec[i] <- "newest assembly"
        full_name[i] <- tmp_name
      }
  
    }
  dna_seq_vec[i] <- dna_seq
  strain_vec[i] <- check2[1,]$infraspecific_name
  date_vec[i] <- check2[1,]$seq_rel_date
  srcdb_vec[i] <- "genbank"
  assembly_level[i] <- check2[1,]$assembly_level
  taxid_vec[i] <- check2[1,]$taxid
  }else{
    source_vec[i] <- NA
    dna_seq_vec[i] <- NA
    strain_vec[i] <- NA
    date_vec[i] <- NA
    srcdb_vec[i] <- NA
    assembly_level[i] <- NA
    full_name[i] <- NA
    taxid_vec[i] <- NA 
  }
}


print(paste0("processed ",i," out of ",length(specs)," genomes"))
}

# remove the missing ones
nmis <- unique(c(which(is.na(dna_seq_vec)),which(is.na(full_name))))
if(length(nmis)==0){
  dna_seq_lst <- as.list(dna_seq_vec)
  #seq_names <- paste0(paste0("SP",1:length(specs))," ",specs," ",gsub("="," ",strain_vec), " ", source_vec, " ", srcdb_vec, " date ", date_vec, ", ",assembly_level)
  seq_names <- full_name
}else{
  dna_seq_lst <- as.list(dna_seq_vec[-nmis])
  #seq_names <- paste0(paste0("SP",1:length(specs))," ",specs," ",gsub("="," ",strain_vec), " ", source_vec, " ", srcdb_vec, " date ", date_vec, ", ",assembly_level)
  seq_names <- full_name
  seq_names <- seq_names[-nmis]
}

kraken_headers <- sapply(seq_along(seq_names), function(i) {
  h <- seq_names[i]
  t <- taxid_vec[i]
  parts <- strsplit(h, " ", fixed = TRUE)[[1]]
  id <- parts[1]
  desc <- if (length(parts) > 1) paste(parts[-1], collapse = " ") else ""
  paste0(id, "|kraken:taxid|", t, if (desc != "") paste0(" ", desc) else "")
}, USE.NAMES = FALSE)


# saving to files
fname <- paste0("/Users/matga374/Desktop/mm_refs_for_dbs/",file_name,"_",seq_len,".fasta")
write.fasta(dna_seq_lst, kraken_headers, fname)

fname <- paste0("/Users/matga374/Desktop/mm_refs_for_dbs/",file_name,"_",seq_len,"_comGenOnly.fasta")
ncg <- which(assembly_level[-nmis] == "Complete Genome")
write.fasta(dna_seq_lst[ncg], kraken_headers[ncg], fname)

#rm(list = ls(all.names = TRUE))
}