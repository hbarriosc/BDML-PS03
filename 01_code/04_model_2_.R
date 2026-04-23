##Prediccion 1.2 LPM Version mejorada

set.seed(123)


#Seleccionamos la variables a vivienda, tamaño del hogar, educación,
#ocupación, formalidad laboral, salud y transferencias y unicamente nos quedarmos con ñas que  en train como en test
#Con el fin de mejorar nuestra prediccion
#Se incluyeron variables de educación y empleo porque, desde la teoría del capital humano, mayores niveles educativos y mejores inserciones
#laborales aumentan la productividad y reducen la probabilidad de pobreza.

vars_economicas <- c(
  # Características del hogar
  "P5000",        # Número de cuartos
  "P5010",        # Cuartos para dormir
  "P5090",        # Tenencia de vivienda
  "Nper",         # Número de personas
  "Npersug",      # Personas en unidad de gasto
  
  # Características promedio de personas en el hogar
  "prom_P6040",   # Edad promedio
  "prom_P6210",   # Nivel educativo (MUY importante teóricamente)
  "prom_P6240",   # Actividad semana pasada (ocupación)
  "prom_P6430",   # Tipo de empleo (cuenta propia, empleado, etc.)
  "prom_P6800",   # Horas trabajadas
  "prom_P6920",   # Cotiza pensión (proxy formalidad)
  "prom_P6090",   # Afiliación salud
  "prom_P7495",   # Recibe arriendos
  "prom_P7505",   # Recibe transferencias
  "prom_Oc",      # Ocupados en el hogar
  "n_personas"    # Total personas
)

#Tomamos las que existan en train y test

vars_disponibles <- intersect(
  intersect(vars_economicas, names(train)),
  intersect(vars_economicas, names(test))
)

cat("Variables seleccionadas:", length(vars_disponibles), "\n")
print(vars_disponibles)


#Construimos las bases 

train_lpm2 <- train %>% select(all_of(c(vars_disponibles, "Pobre")))
test_lpm2  <- test  %>% select(all_of(vars_disponibles))

# Imputación con mediana calculada solo en train

medianas <- sapply(train_lpm2 %>% select(-Pobre), 
                   function(x) median(x, na.rm = TRUE))

for (col in names(medianas)) {
  med <- ifelse(is.na(medianas[col]), 0, medianas[col])
  train_lpm2[[col]][is.na(train_lpm2[[col]])] <- med
  test_lpm2[[col]][is.na(test_lpm2[[col]])]   <- med
}

# Variable objetivo como 0/1
train_lpm2$Pobre_num <- ifelse(train_lpm2$Pobre == "Yes", 1, 0)

cat("NAs en train:", sum(is.na(train_lpm2)), "\n")
cat("NAs en test:", sum(is.na(test_lpm2)), "\n")



# Definimos control con CV de 5 folds
ctrl_cv <- trainControl(
  method          = "cv",
  number          = 5,
  summaryFunction = prSummary,   
  classProbs      = TRUE,
  savePredictions = "final"
)

# Para usar trainControl de caret con LPM usamos lm directamente
# Hacemos CV manual para mayor control

set.seed(123)
folds <- createFolds(train_lpm2$Pobre_num, k = 5, list = TRUE)

f1_folds <- c()

for (i in seq_along(folds)) {
  idx_val   <- folds[[i]]
  fold_train <- train_lpm2[-idx_val, ]
  fold_valid <- train_lpm2[idx_val, ]
  
  # Entrenamos LPM
  formula_lpm <- as.formula(
    paste("Pobre_num ~", paste(vars_disponibles, collapse = " + "))
  )
  
  modelo_fold <- lm(formula_lpm, data = fold_train)
  
  # Predicción
  prob_val <- predict(modelo_fold, newdata = fold_valid)
  prob_val <- pmax(pmin(prob_val, 1), 0)
  
  # F1 con cutoff 0.5
  pred_val <- ifelse(prob_val >= 0.5, 1, 0)
  
  tp <- sum(pred_val == 1 & fold_valid$Pobre_num == 1)
  fp <- sum(pred_val == 1 & fold_valid$Pobre_num == 0)
  fn <- sum(pred_val == 0 & fold_valid$Pobre_num == 1)
  
  prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
  rec  <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
  f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
  
  f1_folds[i] <- f1
  cat(sprintf("Fold %d - F1: %.4f\n", i, f1))
}

cat(sprintf("\nF1 promedio CV: %.4f ± %.4f\n", mean(f1_folds), sd(f1_folds)))


# Manejamos desbalance

prop_pobre <- mean(train_lpm2$Pobre_num)
cat(sprintf("Proporción pobres: %.2f%%\n", prop_pobre * 100))

# Peso inversamente proporcional a la frecuencia de clase
pesos <- ifelse(
  train_lpm2$Pobre_num == 1,
  1 / prop_pobre,         # Peso alto para pobres (minoría)
  1 / (1 - prop_pobre)    # Peso bajo para no pobres (mayoría)
)

# Modelo final con PESOS para corregir desbalance
formula_final <- as.formula(
  paste("Pobre_num ~", paste(vars_disponibles, collapse = " + "))
)

lpm_fit_v2 <- lm(formula_final, data = train_lpm2, weights = pesos)

summary(lpm_fit_v2)


#Se busca el mejor cutoff

cutoffs_cv <- seq(0.10, 0.90, by = 0.05)
f1_por_cutoff <- numeric(length(cutoffs_cv))

for (j in seq_along(cutoffs_cv)) {
  cut <- cutoffs_cv[j]
  f1_temp <- c()
  
  for (i in seq_along(folds)) {
    idx_val    <- folds[[i]]
    fold_train <- train_lpm2[-idx_val, ]
    fold_valid <- train_lpm2[idx_val, ]
    
    pesos_fold <- ifelse(
      fold_train$Pobre_num == 1,
      1 / mean(fold_train$Pobre_num),
      1 / (1 - mean(fold_train$Pobre_num))
    )
    
    mod_temp <- lm(formula_final, data = fold_train, weights = pesos_fold)
    
    prob_temp <- predict(mod_temp, newdata = fold_valid)
    prob_temp <- pmax(pmin(prob_temp, 1), 0)
    pred_temp <- ifelse(prob_temp >= cut, 1, 0)
    
    tp <- sum(pred_temp == 1 & fold_valid$Pobre_num == 1)
    fp <- sum(pred_temp == 1 & fold_valid$Pobre_num == 0)
    fn <- sum(pred_temp == 0 & fold_valid$Pobre_num == 1)
    
    prec <- ifelse((tp + fp) == 0, 0, tp / (tp + fp))
    rec  <- ifelse((tp + fn) == 0, 0, tp / (tp + fn))
    f1   <- ifelse((prec + rec) == 0, 0, 2 * prec * rec / (prec + rec))
    
    f1_temp[i] <- f1
  }
  
  f1_por_cutoff[j] <- mean(f1_temp)
}

resultados_cutoff <- data.frame(
  cutoff = cutoffs_cv,
  f1_cv  = f1_por_cutoff
)

print(resultados_cutoff)

mejor_cutoff_v2 <- resultados_cutoff$cutoff[which.max(resultados_cutoff$f1_cv)]
cat(sprintf("\nMejor cutoff (CV): %.2f | F1 CV: %.4f\n", 
            mejor_cutoff_v2, max(f1_por_cutoff)))


#Validacion final 


prob_test_v2 <- predict(lpm_fit_v2, newdata = test_lpm2)
prob_test_v2 <- pmax(pmin(prob_test_v2, 1), 0)
pred_test_v2 <- ifelse(prob_test_v2 >= mejor_cutoff_v2, 1, 0)

# Imputar NAs residuales (si los hay) con la clase mayoritaria
pred_test_v2[is.na(pred_test_v2)] <- 0
pred_test_v2 <- as.integer(pred_test_v2)

# Diagnóstico
cat(sprintf("Predichos como pobres: %d (%.1f%%)\n",
            sum(pred_test_v2), mean(pred_test_v2) * 100))


# SUBMISSION


submission_lpm_v2 <- data.frame(
  id    = test$id,
  Pobre = pred_test_v2
)

# Validaciones finales
cat("Dimensiones:", dim(submission_lpm_v2), "\n")
cat("NAs en Pobre:", sum(is.na(submission_lpm_v2$Pobre)), "\n")
cat("Valores únicos:", unique(submission_lpm_v2$Pobre), "\n")

write.csv(
  submission_lpm_v2,
  "submission_lpm_v2_mejorado_cv.csv",
  row.names = FALSE,
  quote     = FALSE
)
