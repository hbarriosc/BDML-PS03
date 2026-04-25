###Predicción 4 Logit Mejorado:

#A contunacion realizaremos una segunda predicción del modelo logit con variables socieconomicas que 
#suele estar asociada con menor capital humano, peores condiciones habitacionales, menor inserción 
#laboral formal y mayor presión demográfica sobre los recursos del hogar.

# Partimos de la base común
train_logit2 <- train
test_logit2  <- test


#  Variables con sustento teorico 

train_logit2 <- train_logit2 %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

test_logit2 <- test_logit2 %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )


# Selección de variables 

vars_logit2 <- c(
  "id", "Pobre",
  "P5000", "P5010", "P5090",
  "Nper", "Npersug",
  "prom_P6040",
  "prom_P6090", "prom_P6100",
  "prom_P6210", "prom_P6210s1",
  "prom_P6240", "prom_P6430",
  "prom_P6800", "prom_P6920",
  "prom_P7495", "prom_P7505",
  "prom_Oc",
  "n_personas",
  "hacinamiento", "cuartos_percapita"
)

vars_logit2 <- intersect(vars_logit2, names(train_logit2))

train_logit2 <- train_logit2[, vars_logit2]
test_logit2  <- test_logit2[, setdiff(vars_logit2, "Pobre")]


# Eliminamos id para modelar

train_model2 <- train_logit2 %>% select(-id)
test_model2  <- test_logit2 %>% select(-id)


# Split train-validation
#Aplicamos nuestra semilla
set.seed(123)
idx2 <- createDataPartition(train_model2$Pobre, p = 0.8, list = FALSE)

train_split2 <- train_model2[idx2, ]
valid_split2 <- train_model2[-idx2, ]


#Convertimos algunas variables categóricas
vars_factor2 <- c(
  "P5090", "prom_P6090", "prom_P6100",
  "prom_P6210", "prom_P6240", "prom_P6430",
  "prom_P6920", "prom_P7495", "prom_P7505"
)

vars_factor2 <- intersect(vars_factor2, names(train_split2))

for (v in vars_factor2) {
  train_split2[[v]] <- as.factor(train_split2[[v]])
  valid_split2[[v]] <- as.factor(valid_split2[[v]])
  test_model2[[v]]  <- as.factor(test_model2[[v]])
}


# Imputación:


moda <- function(x) {
  ux <- na.omit(unique(x))
  ux[which.max(tabulate(match(x, ux)))]
}

for (col in names(train_split2)) {
  if (col != "Pobre") {
    
    if (is.numeric(train_split2[[col]])) {
      med <- median(train_split2[[col]], na.rm = TRUE)
      if (is.na(med)) med <- 0
      
      train_split2[[col]][is.na(train_split2[[col]])] <- med
      valid_split2[[col]][is.na(valid_split2[[col]])] <- med
      test_model2[[col]][is.na(test_model2[[col]])]   <- med
      
    } else {
      mo <- moda(train_split2[[col]])
      
      train_split2[[col]][is.na(train_split2[[col]])] <- mo
      valid_split2[[col]][is.na(valid_split2[[col]])] <- mo
      test_model2[[col]][is.na(test_model2[[col]])]   <- mo
      
      train_split2[[col]] <- factor(train_split2[[col]])
      valid_split2[[col]] <- factor(valid_split2[[col]], levels = levels(train_split2[[col]]))
      test_model2[[col]]  <- factor(test_model2[[col]],  levels = levels(train_split2[[col]]))
    }
  }
}


# Verificación de faltantes


sum(is.na(train_split2))
sum(is.na(valid_split2))
sum(is.na(test_model2))


# Entrenamos Logit

logit_fit2 <- glm(Pobre ~ ., data = train_split2, family = binomial)

summary(logit_fit2)


#Predicciones


logit_prob2 <- predict(logit_fit2, newdata = valid_split2, type = "response")
test_prob_logit2 <- predict(logit_fit2, newdata = test_model2, type = "response")


# Función F1


f1_eval <- function(prob, y_true, cutoff = 0.5) {
  pred <- ifelse(prob >= cutoff, "Yes", "No")
  pred <- factor(pred, levels = c("No", "Yes"))
  
  cm <- confusionMatrix(pred, y_true, positive = "Yes")
  
  precision <- cm$byClass["Precision"]
  recall <- cm$byClass["Recall"]
  f1 <- 2 * precision * recall / (precision + recall)
  
  list(
    confusion = cm,
    precision = precision,
    recall = recall,
    f1 = f1
  )
}


#Evaluación inicial

res_logit2 <- f1_eval(logit_prob2, valid_split2$Pobre, cutoff = 0.5)

res_logit2$f1
res_logit2$precision
res_logit2$recall
res_logit2$confusion


#Buscar mejor cutoff

cuts <- seq(0.10, 0.90, by = 0.05)
resultados_logit2 <- data.frame()

for (cut in cuts) {
  met <- f1_eval(logit_prob2, valid_split2$Pobre, cutoff = cut)
  
  resultados_logit2 <- rbind(
    resultados_logit2,
    data.frame(
      cutoff = cut,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_logit2
resultados_logit2[which.max(resultados_logit2$f1), ]

mejor_cutoff_logit2 <- resultados_logit2$cutoff[which.max(resultados_logit2$f1)]
mejor_cutoff_logit2


# Predicción final

test_pred_logit2 <- ifelse(test_prob_logit2 >= mejor_cutoff_logit2, 1, 0)
test_pred_logit2[is.na(test_pred_logit2)] <- 0
test_pred_logit2 <- as.integer(test_pred_logit2)


#Submission

submission_logit2 <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_logit2
)

head(submission_logit2)
dim(submission_logit2)
sum(is.na(submission_logit2$Pobre))
unique(submission_logit2$Pobre)
names(submission_logit2)

write.csv(
  submission_logit2,
  here("03_output/submissions","submission_logit_v2_mejorado.csv"),
  row.names = FALSE,
  quote = FALSE
)
