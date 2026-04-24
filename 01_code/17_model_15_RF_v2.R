###Predicción 15 Random Forest alterno
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

# Convertimos variable objetivo a factor para clasificación
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

##Entrenamiento RANDOM FOREST
set.seed(123)

control <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

#Grid
grid <- expand.grid(
  mtry = c(5, 10), 
  splitrule = "gini", 
  min.node.size = c(1, 5)
)

train_modelado <- train_imp %>% select(-id)

modelo_rf <- train(
  Pobre ~ ., 
  data = train_modelado,
  method = "ranger",
  trControl = control,
  tuneGrid = grid,
  num.trees = 500,
  importance = "impurity"
)

#Predicción
test_modelado <- test_imp %>% select(-id)

pred_prob  <- predict(modelo_rf, newdata = test_modelado, type = "prob")
pred_clase <- predict(modelo_rf, newdata = test_modelado)

# Submission
submission <- data.frame(
  id = test_imp$id,
  Pobre = ifelse(pred_clase == "Si", 1, 0)
)

#Guardamos resultados
write.csv(submission, here("03_output/submissions","predicciones_pobreza_final.csv"), row.names = FALSE)
