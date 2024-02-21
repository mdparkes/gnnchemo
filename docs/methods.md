# Methods

## Data processing

Drug data were downloaded from Genomic Data Commons (GDC) using the R library
`TCGAbiolinks`. All available TCGA cancer types were included. Drug data were
obtained in XML format, which is the format with the most comprehensive data.

Drug data were written to a CSV file. The CSV was manually edited in a spread-
sheet application before importing the preprocessed data back into R for
finishing. Manual preprocessing involved several steps. Typographical errors 
in the drug names were corrected and all drugs were given generic names. 
Combination therapies such as FOLFIRI were listed as their component drugs.
Drugs administered in combination were pipe-delimited. A new column called 
`drug_type` was created to assign each drug to a category according to its
mechanism of action. Categories are pipe-delimited if multiple drugs were
administered in combination, and they are presented in the same order the
drugs are listed in the `drug_name` column. Some drugs belonged to more than
one category. Entries with missing data in the `measure_of_response` column
were removed, as were entries with missing data about the delay between date of
initial pathologic diagnosis and the start and end of therapy. The delay
between initial pathologic diagnosis and biopsy collection is not known, but it
is assumed to be fairly short. Patients were removed if they had indicating a
treatment start date later than one year post-diagnosis were removed because 
these biopsies are unlikely to bear a phenotype that relates to treatment
response. Patients were removed if any of their therapies were recorded as
beginning before diagnosis or beginning and ending on the same day.

Patients in the dataset often underwent multiple phases of combination therapy,
and not all therapies that were administered together were started/ended at the
same time. Considering this in combination with the fact that chemotherapy
alters biopsy phenotypes, the data were filtered to include only the earliest
measure of response to therapy and the combination of drugs that were being
administered at the time that response was measured. 

Response to therapy is an ordinal variable but the number of observations in
some of the classes for certain drug types is too low to reliably model.
Therefore, the measure of response was binarized into `Positive Response` and
`Minimal Response`. The former includes complete and partial responses. The
latter includes patients whose tumors remained stable or progressed under 
treatment. Note that this does not necessarily mean that the treatment had no
effect, but simply that it did not have the intended effect.

The data are insufficient to model responses to specific drugs, which is why the
drugs were categorized according to their primary mechanism of action. Despite
classifying each treatment according to its mechanism, only four types of drugs
were deemed ubiquitous enough for modeling responses: alkylating agents,
antimetabolites, antimitotics, and topoisomerase inhibitors. All other drugs
were grouped under the `Other` category for the purpose of modeling. The
combination of drug types administered at the time response was measured
(including `Other`) is represented as a sequence of bits.
