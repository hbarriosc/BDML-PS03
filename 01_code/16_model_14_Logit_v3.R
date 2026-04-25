###Predicción 14 Logit alterno v3
procesar_base <- function(df_hogares, df_personas) {
  personas_agg <- df_personas %>%
    group_by(id) %>%
    summarise(
      across(where(is.numeric), ~mean(.x, na.rm = TRUE), .names = "prom_{.col}"),
      n_personas = n()
    )
  df_hogares %>% left_join(personas_agg, by = "id")
}

train_full <- procesar_base(train_hogares, train_personas)
test_full  <- procesar_base(test_hogares, test_personas)

# Limpieza de NAs y variables redundantes
cols_keep <- colMeans(is.na(train_full)) <= 0.3
vars_comunes <- intersect(names(train_full[, cols_keep]), names(test_full))
vars_eliminar <- c("prom_Clase", "prom_Fex_c", "prom_Depto", "prom_Fex_dpto")
vars_comunes <- setdiff(vars_comunes, vars_eliminar)

preparar_final <- function(df, es_train = TRUE) {
  df_proc <- df %>%
    select(any_of(c(vars_comunes, "Pobre", "id"))) %>%
    mutate(
      hacinamiento = P5010 / (P5000 + 0.1),
      dependencia  = n_personas / (prom_Oc + 1),
      educacion    = prom_P6210 + (replace_na(prom_P6210s1, 0) / 10),
      Dominio      = as.factor(Dominio)
    )
  if(es_train) df_proc$Pobre <- factor(df_proc$Pobre, levels = c(0, 1), labels = c("No", "Si"))
  return(df_proc)
}

train_df <- preparar_final(train_full, es_train = TRUE)
test_df  <- preparar_final(test_full, es_train = FALSE)
test_df$Dominio <- factor(test_df$Dominio, levels = levels(train_df$Dominio))

# Imputación
prep_imp <- preProcess(train_df %>% select(-id, -Pobre), method = c("medianImpute", "center", "scale"))
train_imp <- predict(prep_imp, train_df)
test_imp  <- predict(prep_imp, test_df)

set.seed(123)

train_logit_input <- train_imp %>% select(-id)

control_logit <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

modelo_logit <- train(
  Pobre ~ ., 
  data = train_logit_input, 
  method = "glm", 
  family = "binomial",
  trControl = control_logit
)

#Predicción
test_logit_input <- test_imp %>% select(-id)

final_preds_logit <- predict(modelo_logit, newdata = test_logit_input)

submission_logit <- data.frame(
  id = test_imp$id, 
  Pobre = ifelse(final_preds_logit == "Si", 1, 0)
)

write.csv(submission_logit, here("03_output/submissions","predicciones_pobreza_logit.csv")"", row.names = FALSE)