###Predicción 13 ELASTIC NET Tuning de Alpha y Lambda
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

# Identificamos variables con > 30% de NAs en Train para eliminarlas en ambos
cols_con_pocos_nas <- colMeans(is.na(train_full)) <= 0.3
train_clean <- train_full[, cols_con_pocos_nas]

# Sincronizar columnas comunes
vars_comunes <- intersect(names(train_clean), names(test_full))
vars_eliminar <- c("prom_Clase", "prom_Fex_c", "prom_Depto", "prom_Fex_dpto")
vars_comunes <- setdiff(vars_comunes, vars_eliminar)

# Bases filtradas
train_final <- train_clean[, c(vars_comunes, "Pobre")]
test_final  <- test_full[, vars_comunes]

#Ajuste de variables
preparar_datos <- function(df) {
  df %>%
    mutate(
      hacinamiento = P5010 / (P5000 + 0.1),
      dependencia  = n_personas / (prom_Oc + 1),
      educacion    = prom_P6210 + (replace_na(prom_P6210s1, 0) / 10),
      horas_total  = prom_P6800 * n_personas,
      Dominio      = as.factor(Dominio)
    )
}

train_fe <- preparar_datos(train_final)
test_fe  <- preparar_datos(test_final)

#convertimos la variable objetivo a factor para clasificación
train_fe$Pobre <- factor(train_fe$Pobre, levels = c(0, 1), labels = c("No", "Si"))

test_fe$Dominio <- factor(test_fe$Dominio, levels = levels(train_fe$Dominio))

#Imputación
na_vars <- names(train_fe)[colSums(is.na(train_fe)) > 0]

for(v in na_vars) {
  train_fe[[paste0(v, "_NA")]] <- as.numeric(is.na(train_fe[[v]]))
  test_fe[[paste0(v, "_NA")]]  <- as.numeric(is.na(test_fe[[v]]))
}

prep_mediana <- preProcess(train_fe %>% select(-id, -Pobre), method = "medianImpute")

train_imp <- predict(prep_mediana, train_fe)
test_imp  <- predict(prep_mediana, test_fe)

# Escalación de datos
prep_en <- preProcess(train_imp %>% select(-id, -Pobre), method = c("center", "scale", "medianImpute"))

train_en <- predict(prep_en, train_imp)
test_en  <- predict(prep_en, test_imp)

#Entrenamiento ELASTIC NET 

set.seed(123)

#Grilla
en_grid <- expand.grid(
  alpha = seq(0, 1, length = 5), 
  lambda = exp(seq(-10, -1, length = 20))
)

control_en <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

# Entrenamos
train_en_input <- train_en %>% select(-id)

modelo_en <- train(
  Pobre ~ ., 
  data = train_en_input,
  method = "glmnet", 
  family = "binomial",
  trControl = control_en,
  tuneGrid = en_grid,
  metric = "ROC"
)

#Predicción
test_en_input <- test_en %>% select(-id)
final_preds_en <- predict(modelo_en, newdata = test_en_input)

submission_en <- data.frame(
  id = test_en$id, 
  Pobre = ifelse(final_preds_en == "Si", 1, 0)
)

write.csv(submission_en, here("03_output/submissions","predicciones_pobreza_en.csv"), row.names = FALSE)