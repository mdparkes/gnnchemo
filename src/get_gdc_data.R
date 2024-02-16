# libraries ####
library(dplyr)
library(stringr)
library(TCGAbiolinks)

# Functions ####
get_drug_data <- function(cancer) {
  project <- paste0("TCGA-", cancer)
  qry <- GDCquery(
    project,
    data.category = "Clinical", 
    data.format = "bcr xml",
  )
  GDCdownload(qry)
  GDCprepare_clinic(qry, clinical.info = "drug")
}

subset_drug_data <- function(x) {
  x[x==""] <- NA
  sel_rows <- is.na(x$drug_name) | is.na(x$measure_of_response)
  # sel_cols <- c(
  #   "project",
  #   "bcr_patient_barcode",
  #   "therapy_types",
  #   "drug_name",
  #   "measure_of_response",
  #   "days_to_drug_therapy_start",
  #   "days_to_drug_therapy_end"
  # )
  x[!sel_rows, ]
}

# Get GDC clinical data ####
# tcga_types <- c(
#   "BLCA",
#   "BRCA",
#   "COAD",
#   "GBM",
#   "LIHC",
#   "LUAD",
#   "LUSC",
#   "STAD",
#   "UCEC"
# )

tcga_types <- c(
  "ACC",
  "BLCA",
  "BRCA",
  "CESC",
  "CHOL",
  "COAD",
  "ESCA",
  "HNSC",
  "KICH",
  "KIRC",
  "KIRP",
  "LIHC",
  "LUAD",
  "LUSC",
  "MESO",
  "OV",
  "PAAD",
  "PCPG",
  "PRAD",
  "READ",
  "SARC",
  "SKCM",
  "STAD",
  "TGCT",
  "THCA",
  "THYM",
  "UCEC",
  "UCS",
  "UVM"
)

drug_data_list <- lapply(tcga_types, get_drug_data)
drug_df <- as.data.frame(do.call(rbind, drug_data_list))
write.csv(drug_df, "/Users/pr3/Projects/gnnchemo/data/big_drug_df.csv")
# write.csv(drug_df, "/Users/pr3/Projects/gnnchemo/data/drug_df.csv", sep = ",")
