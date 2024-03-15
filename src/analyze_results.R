library(caret)
library(ROCR)


load_data <- function(model_type, drug_input, train_data = F) {
  # Import a data frame of model predictions
  exp_pth <- "experiment1/exports/"
  use_drug <- ifelse(drug_input, "True", "False")
  base_name <- paste0(exp_pth, model_type, "/",
                      "model-type=", model_type,
                      "_drug-input=", use_drug)
  if (train_data) {
    file_name <- paste0(base_name, "_train_predictions_df.csv")
  } else {
    file_name <- paste0(base_name, "_test_predictions_df.csv")
  }
  pred_df <- read.csv(file_name, row.names = 1)
  row.names(pred_df) <- vapply(
    row.names(pred_df),
    function(id) sub(".pt", "", id),
    FUN.VALUE = character(1L), USE.NAMES = F
  )
  pred_df$predicted_class <- ifelse(pred_df$prediction < 0.5, 0, 1)
  colnames(pred_df) <- c("probability", "label", "prediction")
  pred_df <- pred_df[, c(1, 3, 2)]
  return(pred_df)
}

# Import model predictions
gnn_test_df_drug <- load_data("gnn", drug_input = T, train_data = F)
gnn_train_df_drug <- load_data("gnn", drug_input = T, train_data = T)
gnn_test_df_no_drug <- load_data("gnn", drug_input = F, train_data = F)
gnn_train_df_no_drug <- load_data("gnn", drug_input = F, train_data = T)

mlp_test_df_drug <- load_data("mlp", drug_input = T, train_data = F)
mlp_train_df_drug <- load_data("mlp", drug_input = T, train_data = T)
mlp_test_df_no_drug <- load_data("mlp", drug_input = F, train_data = F)
mlp_train_df_no_drug <- load_data("mlp", drug_input = F, train_data = T)

# Performance ####

# Test sets
gnn_test_pred_drug <- prediction(gnn_test_df_drug$prediction,
                                 gnn_test_df_drug$label)
gnn_test_pred_no_drug <- prediction(gnn_test_df_no_drug$prediction,
                                    gnn_test_df_no_drug$label)
mlp_test_pred_drug <- prediction(mlp_test_df_drug$prediction,
                                 mlp_test_df_drug$label)
mlp_test_pred_no_drug <- prediction(mlp_test_df_no_drug$prediction,
                                    mlp_test_df_no_drug$label)

gnn_test_prec_drug <- performance(gnn_test_pred_drug, "prec")
gnn_test_prec_no_drug <- performance(gnn_test_pred_no_drug, "prec")
mlp_test_prec_drug <- performance(mlp_test_pred_drug, "prec")
mlp_test_prec_no_drug <- performance(mlp_test_pred_no_drug, "prec")

gnn_test_rec_drug <- performance(gnn_test_pred_drug, "rec")
gnn_test_rec_no_drug <- performance(gnn_test_pred_no_drug, "rec")
mlp_test_rec_drug <- performance(mlp_test_pred_drug, "rec")
mlp_test_rec_no_drug <- performance(mlp_test_pred_no_drug, "rec")

gnn_test_pr_drug <- performance(gnn_test_pred_drug, "prec", "rec")
gnn_test_pr_no_drug <- performance(gnn_test_pred_no_drug, "prec", "rec")
mlp_test_pr_drug <- performance(mlp_test_pred_drug, "prec", "rec")
mlp_test_pr_no_drug <- performance(mlp_test_pred_no_drug, "prec", "rec")

gnn_test_roc_drug <- performance(gnn_test_pred_drug, "tpr", "fpr")
gnn_test_roc_no_drug <- performance(gnn_test_pred_no_drug, "tpr", "fpr")
mlp_test_roc_drug <- performance(mlp_test_pred_drug, "tpr", "fpr")
mlp_test_roc_no_drug <- performance(mlp_test_pred_no_drug, "tpr", "fpr")

gnn_test_f_drug <- performance(gnn_test_pred_drug, "f", alpha = 0.5)
gnn_test_f_no_drug <- performance(gnn_test_pred_no_drug, "f", alpha = 0.5)
mlp_test_f_drug <- performance(mlp_test_pred_drug, "f", alpha = 0.5)
mlp_test_f_no_drug <- performance(mlp_test_pred_no_drug, "f", alpha = 0.5)

gnn_test_auc_drug <- performance(gnn_test_pred_drug, "auc")
gnn_test_auc_no_drug <- performance(gnn_test_pred_no_drug, "auc")
mlp_test_auc_drug <- performance(mlp_test_pred_drug, "auc")
mlp_test_auc_no_drug <- performance(mlp_test_pred_no_drug, "auc")

gnn_test_cm_drug <- caret::confusionMatrix(
  table(factor(gnn_test_df_no_drug$prediction,
               levels = c(1, 0), 
               labels = c("Positive response", "Minimal response")),
        factor(gnn_test_df_no_drug$label,
               levels = c(1, 0), 
               labels = c("Positive response", "Minimal response")),
        dnn = list("Predicted", "Actual")
  ),
  positive = "Positive response"
)


# Training sets
gnn_train_pred_drug <- prediction(gnn_train_df_drug$prediction,
                                 gnn_train_df_drug$label)
gnn_train_pred_no_drug <- prediction(gnn_train_df_no_drug$prediction,
                                    gnn_train_df_no_drug$label)
mlp_train_pred_drug <- prediction(mlp_train_df_drug$prediction,
                                 mlp_train_df_drug$label)
mlp_train_pred_no_drug <- prediction(mlp_train_df_no_drug$prediction,
                                    mlp_train_df_no_drug$label)

gnn_train_prec_drug <- performance(gnn_train_pred_drug, "prec")
gnn_train_prec_no_drug <- performance(gnn_train_pred_no_drug, "prec")
mlp_train_prec_drug <- performance(mlp_train_pred_drug, "prec")
mlp_train_prec_no_drug <- performance(mlp_train_pred_no_drug, "prec")

gnn_train_rec_drug <- performance(gnn_train_pred_drug, "rec")
gnn_train_rec_no_drug <- performance(gnn_train_pred_no_drug, "rec")
mlp_train_rec_drug <- performance(mlp_train_pred_drug, "rec")
mlp_train_rec_no_drug <- performance(mlp_train_pred_no_drug, "rec")

gnn_train_pr_drug <- performance(gnn_train_pred_drug, "prec", "rec")
gnn_train_pr_no_drug <- performance(gnn_train_pred_no_drug, "prec", "rec")
mlp_train_pr_drug <- performance(mlp_train_pred_drug, "prec", "rec")
mlp_train_pr_no_drug <- performance(mlp_train_pred_no_drug, "prec", "rec")

gnn_train_roc_drug <- performance(gnn_train_pred_drug, "tpr", "fpr")
gnn_train_roc_no_drug <- performance(gnn_train_pred_no_drug, "tpr", "fpr")
mlp_train_roc_drug <- performance(mlp_train_pred_drug, "tpr", "fpr")
mlp_train_roc_no_drug <- performance(mlp_train_pred_no_drug, "tpr", "fpr")

gnn_train_f_drug <- performance(gnn_train_pred_drug, "f", alpha = 0.5)
gnn_train_f_no_drug <- performance(gnn_train_pred_no_drug, "f", alpha = 0.5)
mlp_train_f_drug <- performance(mlp_train_pred_drug, "f", alpha = 0.5)
mlp_train_f_no_drug <- performance(mlp_train_pred_no_drug, "f", alpha = 0.5)

# Plots
plot(gnn_test_pr_drug, avg = "threshold", colorize=TRUE, lwd= 3)
plot(mlp_test_pr_drug, avg = "threshold", colorize=TRUE, lwd= 3)
plot(mlp_test_pr_no_drug, avg = "threshold", colorize=TRUE, lwd= 3)

plot(gnn_test_roc_drug, avg = "threshold", colorize=TRUE, lwd= 3)
plot(mlp_test_roc_drug, avg = "threshold", colorize=TRUE, lwd= 3)
plot(mlp_test_roc_no_drug, avg = "threshold", colorize=TRUE, lwd= 3)
