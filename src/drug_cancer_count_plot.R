# Credit: https://www.data-to-viz.com/graph/circularbarplot.html

library(dplyr)
library(ggplot2)
library(stringr)

project_dir <- "/Users/pr3/Projects/gnnchemo"
# Load original data from csv
file_in <- paste0(project_dir, "/data/drug_df.csv")
drug_df <- read.csv(file_in)[ ,-1]

# Load cleaned data from csv
file_in <- paste0(project_dir, "/data/processed_drug_df.csv")
processed_drug_df <- read.csv(file_in)[ ,-1]

# Get TCGA cancer type for entries in processed_drug_df
bcr_patient_barcode <- vapply(
  processed_drug_df$aliquot_submitter_id,
  function(x) str_extract(x, "TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}"),
  FUN.VALUE = character(1), USE.NAMES = F
)
m <- pmatch(bcr_patient_barcode, drug_df$bcr_patient_barcode)
processed_drug_df$project <- drug_df$project[m]
processed_drug_df$project <- vapply(
  processed_drug_df$project,
  function (x) str_extract(x, "[A-Z]{4}$"),
  FUN.VALUE = character(1), USE.NAMES = F
)

alkylating <- processed_drug_df$project[processed_drug_df$alkylating_agent]
antimetabolite <- processed_drug_df$project[processed_drug_df$antimetabolite]
antimitotic <- processed_drug_df$project[processed_drug_df$antimitotic]
topoisomerase <- processed_drug_df$project[processed_drug_df$topoisomerase_inhibitor]
other <- processed_drug_df$project[processed_drug_df$other_drug]

alkylating_table <- table(alkylating)
antimetabolite_table <- table(antimetabolite)
antimitotic_table <- table(antimitotic)
topoisomerase_table <- table(topoisomerase)
other_table <- table(other)

data <- data.frame(
  Drug = c(
    rep("Alkylating agent", length(alkylating_table)),
    rep("Antimetabolite", length(antimetabolite_table)),
    rep("Antimitotic", length(antimitotic_table)),
    rep("Topoisomerase inhibitor", length(topoisomerase_table)),
    rep("Other drug", length(other_table))
  ),
  Cancer = c(
    paste(names(alkylating_table), alkylating_table),
    paste(names(antimetabolite_table), antimetabolite_table),
    paste(names(antimitotic_table), antimitotic_table),
    paste(names(topoisomerase_table), topoisomerase_table),
    paste(names(other_table), other_table)
  ),
  Count = c(
    alkylating_table,
    antimetabolite_table,
    antimitotic_table,
    topoisomerase_table,
    other_table
  )
)
data$Drug <- factor(data$Drug, 
                    levels = c(
                      "Alkylating agent",
                      "Antimetabolite",
                      "Antimitotic",
                      "Topoisomerase inhibitor",
                      "Other drug"
                    ))
data <- data %>% arrange(Drug, Count)

# Set a number of 'empty bar' to add at the end of each group
empty_bar <- 3
to_add <- data.frame(matrix(NA, empty_bar * nlevels(data$Drug), ncol(data)))
colnames(to_add) <- colnames(data)
to_add$Drug <- rep(levels(data$Drug), each = empty_bar)
data <- rbind(data, to_add)
data <- data %>% arrange(Drug)
data$id <- seq(1, nrow(data))

# Get the name and the y position of each label
label_data <- data
number_of_bar <- nrow(label_data)
angle <- 90 - 360 * (label_data$id - 0.5) / number_of_bar  # Substract 0.5 because the letter must have the angle of the center of the bars. Not extreme right(1) or extreme left (0)
label_data$hjust <- ifelse( angle < -90, 1, 0)
label_data$angle <- ifelse(angle < -90, angle + 180, angle)

# prepare a data frame for base lines
base_data <- data %>% 
  group_by(Drug) %>% 
  summarize(start=min(id), end=max(id) - empty_bar) %>% 
  rowwise() %>% 
  mutate(title = mean(c(start, end)))

# prepare a data frame for grid (scales)
grid_data <- base_data
grid_data$end <- grid_data$end[c(nrow(grid_data), 1:nrow(grid_data) - 1)] + 1
grid_data$start <- grid_data$start - 1
grid_data <- grid_data[-1,]

# Make the plot
p <- ggplot(data, aes(x = as.factor(id), y = Count, fill = Drug)) +       # Note that id is a factor. If x is numeric, there is some space between the first bar
  
  geom_bar(aes(x = as.factor(id), y = Count, fill = Drug), 
           stat = "identity", alpha = 0.5) +
  # Add a val=100/75/50/25 lines. I do it at the beginning to make sur barplots are OVER it.
  geom_segment(data = grid_data,
               aes(x = end, y = 80, xend = start, yend = 80),
               colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE) +
  geom_segment(data = grid_data, 
               aes(x = end, y = 60, xend = start, yend = 60), 
               colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE ) +
  geom_segment(data = grid_data,
               aes(x = end, y = 40, xend = start, yend = 40), 
               colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE ) +
  geom_segment(data = grid_data, 
               aes(x = end, y = 20, xend = start, yend = 20), 
               colour = "grey", alpha = 1, size = 0.3, inherit.aes = FALSE ) +
  # Add text showing the value of each 100/75/50/25 lines
  annotate("text", 
           x = rep(max(data$id), 4), 
           y = c(20, 40, 60, 80), 
           label = c("20", "40", "60", "80"),
           color = "grey", size = 3 , angle = 0, fontface = "bold", hjust = 1) +
  geom_bar(aes(x = as.factor(id), y = Count, fill = Drug), 
           stat = "identity", alpha = 0.5) +
  ylim(-100, 120) +
  theme_minimal() +
  theme(
    legend.position = "none",
    axis.text = element_blank(),
    axis.title = element_blank(),
    panel.grid = element_blank(),
    plot.margin = unit(rep(-1, 4), "cm") 
  ) +
  coord_polar() + 
  geom_text(data = label_data,
            aes(x = id, y = Count + 10, label = Cancer, hjust = hjust),
            color = "black", fontface = "bold", alpha = 0.6, size = 2.5,
            angle = label_data$angle, inherit.aes = FALSE) +
  # Add base line information
  geom_segment(data = base_data,
               aes(x = start, y = -5, xend = end, yend = -5),
               colour = "black", alpha = 0.8, size = 0.6, inherit.aes = FALSE) +
  geom_text(data = base_data,
            aes(x = title, y = -15, label = Drug),
            hjust = c(1, 1, 0.25, 0, 0),
            colour = "black", alpha = 0.8, size = 3, 
            fontface = "bold", inherit.aes = FALSE)

p
