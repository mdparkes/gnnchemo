project_dir <- "/Users/pr3/Projects/gnnchemo"
# Load cleaned data from csv
file_in <- paste0(project_dir, "/data/big_drug_df.csv")
drug_df <- read.csv(file_in)[ ,-1]
# Tabulate response, drug class
table(drug_df$drug_type, drug_df$measure_of_response,
      dnn=list("Drug class", "Response to treatment"))
# Ordinal response to treatment:
#   1: Minimal Response (Stable Disease or Clinical Progressive Disease)
#   2: Partial Response
#   3: Complete Response
minimal <- c("Stable Disease", "Clinical Progressive Disease")
sel_minimal <- drug_df$measure_of_response %in% minimal
drug_df$measure_of_response[sel_minimal] <- "Minimal Response"
# Tabulate response, drug class
table(drug_df$drug_type, drug_df$measure_of_response,
      dnn=list("Drug class", "Response to treatment"))

# Sort the data records by patient barcode, breaking ties by treatment
# end time and then by start time
o <- order(drug_df$bcr_patient_barcode, drug_df$days_to_drug_therapy_end,
           drug_df$days_to_drug_therapy_start, method="radix")
drug_df <- drug_df[o, ]
# Remove patients with missing start or end times in any of their records
sel1 <- is.na(drug_df$days_to_drug_therapy_start)
sel2 <- is.na(drug_df$days_to_drug_therapy_end)
sel <- which(sel1 | sel2)
excl_ids <- unique(drug_df$bcr_drug_barcode[sel])
excl <- which(drug_df$bcr_drug_barcode %in% excl_ids)
drug_df <- drug_df[-excl, ]
# Exclude entries where treatment start time and end time are identical.
# These are probably erroneous since chemo is generally administered over time.
excl <- drug_df$days_to_drug_therapy_start == drug_df$days_to_drug_therapy_end
excl <- which(excl)
drug_df <- drug_df[-excl, ]
# Concatenate treatments administered concurrently under a single entry
# The end result is a pipe-delimited string of all the drugs/drug types that 
# were being administered at each of a patient's treatment end times. For a
# given drug's end time, any other drugs that were terminated sooner will not be
# present in the pipe-delimited string for the given drug even if they were 
# taken together at some point. After the therapies have been concatenated, the
# start times are not necessarily valid for their combination.
excl <- integer(0L)
N <- nrow(drug_df)
same_patient <- duplicated(drug_df$bcr_patient_barcode)
i <- 1L
while (i < N) {
  # Treatment start and end times for patient of entry i
  st_i <- drug_df$days_to_drug_therapy_start[i]
  et_i <- drug_df$days_to_drug_therapy_end[i]
  j <- i + 1L  # Next entry for comparison
  while (j <= N && same_patient[j]) {
    st_j <- drug_df$days_to_drug_therapy_start[j]
    et_j <- drug_df$days_to_drug_therapy_end[j]
    if (st_j < et_i) { # Treatment j starts before i ends
      # Concatenate drug names
      dn_i <- drug_df$drug_name[i]
      dn_j <- drug_df$drug_name[j]
      drug_df$drug_name[i] <- paste(dn_i, dn_j, sep = "|")
      # Concatenate drug types
      dt_i <- drug_df$drug_type[i]
      dt_j <- drug_df$drug_type[j]
      drug_df$drug_type[i] <- paste(dt_i, dt_j, sep = "|")
      dt_i <- drug_df$drug_type_2[i]
      dt_j <- drug_df$drug_type_2[j]
      drug_df$drug_type_2[i] <- paste(dt_i, dt_j, sep = "|")
    }
    if (et_i == et_j) { # Redundant end times; slate row for removal
      excl <- c(excl, j)
    }
    j <- j + 1L
  }
  i <- i + 1L
}
# Remove the redundant entries
drug_df <- drug_df[-excl, ]
# Tabulate response, drug class
table(drug_df$drug_type_2, drug_df$measure_of_response,
      dnn=list("Drug type", "Response to treatment"))
# Select the first endpoint for each patient
sel <- !duplicated(drug_df$bcr_patient_barcode)
drug_df <- drug_df[sel, ]
# Only consider entries that include Alkylating agent, Antimetabolite, or 
# Topoisomerase inhibitor
sel1 <- vapply(
  drug_df$drug_type, grepl, pattern="Alkylating agent",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
sel2 <- vapply(
  drug_df$drug_type, grepl, pattern="Antimetabolite",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
sel3 <- vapply(
  drug_df$drug_type, grepl, pattern="Antimitotic",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
sel4 <- vapply(
  drug_df$drug_type, grepl, pattern="Topoisomerase inhibitor",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
sel <- sel1 | sel2 | sel3 | sel4
# Only include the drug types above
drug_df <- drug_df[sel, ]

# Create columns for each drug type
drug_df$alkylating_agent <- vapply(
  drug_df$drug_type_2, grepl, pattern="Alkylating agent",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
drug_df$antimetabolite <-  vapply(
  drug_df$drug_type_2, grepl, pattern="Antimetabolite",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
drug_df$antimitotic <- vapply(
  drug_df$drug_type_2, grepl, pattern="Antimitotic",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
drug_df$topoisomerase_inhibitor <- vapply(
  drug_df$drug_type_2, grepl, pattern="Topoisomerase inhibitor",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
drug_df$other_drug <- vapply(
  drug_df$drug_type_2, grepl, pattern="Other",
  FUN.VALUE = logical(1L), USE.NAMES = F
)
# Tabulate response, drug class
table(drug_df$drug_type_2, drug_df$measure_of_response,
      dnn=list("Drug type", "Response to treatment"))
# Summary of days to treatment start
summary(drug_df$days_to_drug_therapy_start)
# Only include biopsies from patients whose therapy began within 1 year of dx
excl <- which(is.na(drug_df$days_to_drug_therapy_start))
if (!identical(excl, integer(0L))) {drug_df <- drug_df[-excl, ]}
sel1 <- drug_df$days_to_drug_therapy_start <= 365
sel2 <- drug_df$days_to_drug_therapy_start >= 0
sel <- sel1 & sel2
drug_df <- drug_df[sel, ]  # 1080 biopsies remain
# Tables: Response to therapy vs. cancer type for each drug type
tab <- table(
  drug_df[drug_df$alkylating_agent, "project"],
  drug_df[drug_df$alkylating_agent, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$alkylating_agent, "project"],
  drug_df[drug_df$alkylating_agent, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$antimetabolite, "project"],
  drug_df[drug_df$antimetabolite, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$antimitotic, "project"],
  drug_df[drug_df$antimitotic, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$topoisomerase_inhibitor, "project"],
  drug_df[drug_df$topoisomerase_inhibitor, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

# Only consider the first course of chemotherapy
excl <- which(duplicated(drug_df$bcr_patient_barcode))
if (!identical(excl, integer(0L))) {drug_df <- drug_df[-excl, ]}

# Group partial responders and complete responders
drug_df$measure_of_response <- ifelse(
  drug_df$measure_of_response %in% c("Complete Response", "Partial Response"),
  "Positive response", "Minimal response"
)
# Tables: Response to therapy vs. cancer type for each drug type
tab <- table(
  drug_df[drug_df$alkylating_agent, "project"],
  drug_df[drug_df$alkylating_agent, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$antimetabolite, "project"],
  drug_df[drug_df$antimetabolite, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$antimitotic, "project"],
  drug_df[drug_df$antimitotic, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

tab <- table(
  drug_df[drug_df$topoisomerase_inhibitor, "project"],
  drug_df[drug_df$topoisomerase_inhibitor, "measure_of_response"]
)
rbind(tab, apply(tab, 2, sum)) # Last row is column totals

# Check that gene expression records exist for all drug records
# GitHub can't host the large gene expression file
file_in <- "/Users/pr3/Projects/gnnsurvival-pyg/data/tcga_exprs.csv"
biopsy_ids <- setNames(unlist(data.table::fread(file_in, select = 1)), NULL)
# Exclude drug records that don't match a biopsy
excl <- which(is.na(pmatch(drug_df$bcr_patient_barcode, biopsy_ids)))
drug_df <- drug_df[-excl, ]  # 1001 biopsies remain
sel <- pmatch(drug_df$bcr_patient_barcode, biopsy_ids)
drug_df$aliquot_submitter_id <- biopsy_ids[sel]

# Strip "TCGA-" from drug_df$project
drug_df$project <- vapply(
  drug_df$project,
  gsub, pattern = "^TCGA-", replacement = "",
  FUN.VALUE = character(1L), USE.NAMES = F
)

# Save a compact version of the drug data for modeling
file_out <- paste0(project_dir, "/data/processed_drug_df.csv")
sel_cols <- c("project", "aliquot_submitter_id", "drug_name", "drug_type_2", 
              "alkylating_agent", "antimetabolite", "antimitotic",
              "topoisomerase_inhibitor", "other_drug", "measure_of_response")
obj_out <- drug_df[ ,sel_cols]
write.csv(obj_out, file = file_out, row.names = F)

# Notes: ####
# "Partial Response" tends to be rare. There
# may be too few of them to train a high-quality ordinal regression. The 
# response to treatment could be binarized by putting the Partial Response class
# into either the positive response or the negative response class. It should 
# probably be grouped into the positive responders class so as not to rule out
# what might be a partially effective drug in the absence of alternatives that
# are estimated to be better.
#
# There is a significant imbalance among the responses to treatment between the
# different types of cancer. Consider weighing individual losses according to
# the inverse empirical likelihood of observing (cancer type, response measure).
# This is to avoid a model that simply learns to associate response to specific
# treatment types with a particular type of cancer where it is most often used.
# One could also make an opposing argument: if each cancer origin is outnumbered
# by the collective of all other cancer origins, the model may be discouraged
# from learning shortcuts from tissue signature to treatment effect.
#
# Patients who responded at least partially to treatment outnumber those who 
# responded minimally by roughly four-fold. Weight the losses according to the
# balance of responders and non-responders.
#
# Another limitation is the number of days to start of therapy. If the treatment
# is started before biopsy, it may affect the phenotype. If the treatment is 
# started long after the biopsy was taken, the biopsy's phenotype may not 
# resemble the phenotype at the start of treatment. To complicate matters
# further, there isn't information about the delay between date of initial
# pathologic diagnosis and date of biopsy. It is assumed that the date of biopsy
# is fairly close to the date of initial pathologic diagnosis.
