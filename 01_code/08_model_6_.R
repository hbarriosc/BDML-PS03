#Prediccion 6 Modelo Elastic Net 

#Acontinuacion realizaremos un ejercicio de transformació/y/o depuracion, hay que tener encuenta que la transformacio puede jugar
#a favor o encuentra. pero el ejercicio se basa con el fin de mejorar nuestro modelo anterior
#Validamos nuestras bases iniciales
train_ref <- train
test_ref  <- test

#Validamos las variables exitentes#

var_ingreso_pc <- grep("^ingpcug$|^Ingpcug$|ingpc", names(train_ref), value = TRUE)
var_ingreso_total <- grep("^ingtotug$|^Ingtotug$|ingtot", names(train_ref), value = TRUE)

#Aca buscamos la variable de ingreso con el fin de realziar la trasformacion 

if (length(var_ingreso_pc) > 0) {
  var_ingreso_pc <- var_ingreso_pc[1]
} else {
  var_ingreso_pc <- NA
}

if (length(var_ingreso_total) > 0) {
  var_ingreso_total <- var_ingreso_total[1]
} else {
  var_ingreso_total <- NA
}


#####Escogimos las siguientes variables, con el fin de validar número de cuartos por separado, ya que es una variable
#que puede incurrir en relación con si el hogar es pobre 

train_ref <- train_ref %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

test_ref <- test_ref %>%
  mutate(
    hacinamiento = ifelse(P5010 > 0, Nper / P5010, NA),
    cuartos_percapita = ifelse(Nper > 0, P5000 / Nper, NA)
  )

####Realizamos logartimo a las variables, con el fin de eliminar ruido y que queden mucho mas limpias a la hora de realizar la prediccion
##

if (!is.na(var_ingreso_pc)) {
  train_ref$ingreso_pc_log <- ifelse(train_ref[[var_ingreso_pc]] >= 0,
                                     log1p(train_ref[[var_ingreso_pc]]), NA)
  test_ref$ingreso_pc_log  <- ifelse(test_ref[[var_ingreso_pc]] >= 0,
                                     log1p(test_ref[[var_ingreso_pc]]), NA)
}



if (!is.na(var_ingreso_total)) {
  train_ref$ingreso_total_log <- ifelse(train_ref[[var_ingreso_total]] >= 0,
                                        log1p(train_ref[[var_ingreso_total]]), NA)
  test_ref$ingreso_total_log  <- ifelse(test_ref[[var_ingreso_total]] >= 0,
                                        log1p(test_ref[[var_ingreso_total]]), NA)
}


##### eliminamos las varibles que no utilizaremos### 

train_ref <- train_ref %>% select(-id, -Dominio, -Depto)
test_ref  <- test_ref %>% select(-id, -Dominio, -Depto)

#Vamos a organizar el modelo para el nuevo entranamiento 

set.seed(123)
idx_ref <- createDataPartition(train_ref$Pobre, p = 0.8, list = FALSE)

train_split_ref <- train_ref[idx_ref, ]
valid_split_ref <- train_ref[-idx_ref, ]

train_num_ref <- train_split_ref %>% select(where(is.numeric))
valid_num_ref <- valid_split_ref %>% select(where(is.numeric))
test_num_ref  <- test_ref %>% select(where(is.numeric))


train_num_ref$Pobre <- train_split_ref$Pobre
valid_num_ref$Pobre <- valid_split_ref$Pobre

vars_modelo_ref <- intersect(names(train_num_ref), names(test_num_ref))
vars_modelo_ref <- setdiff(vars_modelo_ref, "Pobre")

train_num_ref <- train_num_ref[, c(vars_modelo_ref, "Pobre")]
valid_num_ref <- valid_num_ref[, c(vars_modelo_ref, "Pobre")]
test_num_ref  <- test_num_ref[, vars_modelo_ref]

### Aqui vamos a utilizar nuvamente la mediana de las varibales para calcular nuestro modelo

for (col in names(train_num_ref)) {
  if (col != "Pobre") {
    mediana_col <- median(train_num_ref[[col]], na.rm = TRUE)
    
    if (is.na(mediana_col)) mediana_col <- 0
    
    train_num_ref[[col]][is.na(train_num_ref[[col]])] <- mediana_col
    valid_num_ref[[col]][is.na(valid_num_ref[[col]])] <- mediana_col
    
    if (col %in% names(test_num_ref)) {
      test_num_ref[[col]][is.na(test_num_ref[[col]])] <- mediana_col
    }
  }
}

#Verificamos que todo este correcto 

sum(is.na(train_num_ref))
sum(is.na(valid_num_ref))
sum(is.na(test_num_ref))


train_num_ref$Pobre <- factor(train_num_ref$Pobre, levels = c("No", "Yes"))
valid_num_ref$Pobre <- factor(valid_num_ref$Pobre, levels = c("No", "Yes"))


nzv_cols <- nearZeroVar(train_num_ref %>% select(-Pobre))

if (length(nzv_cols) > 0) {
  cols_quitar <- names(train_num_ref %>% select(-Pobre))[nzv_cols]
  
  train_num_ref <- train_num_ref %>% select(-all_of(cols_quitar))
  valid_num_ref <- valid_num_ref %>% select(-all_of(cols_quitar))
  test_num_ref  <- test_num_ref %>% select(-all_of(cols_quitar))
}

#Aqui realizamos la mezcla entre ridge  y lasso , con el fin de buscar la mejor combinancion para nuestro modelo 

grid_enet <- expand.grid(
  alpha = seq(0, 1, by = 0.2),
  lambda = 10^seq(-3, 0, length = 20)
)


#Empezamos a entrenar el modelo 

set.seed(123)

enet_fit_ref <- train(
  Pobre ~ .,
  data = train_num_ref,
  method = "glmnet",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = grid_enet
)


enet_fit_ref
enet_fit_ref$bestTune

enet_prob_ref <- predict(enet_fit_ref, newdata = valid_num_ref, type = "prob")[, "Yes"]


res_enet_ref <- f1_eval(enet_prob_ref, valid_num_ref$Pobre, cutoff = 0.5)

res_enet_ref$f1
res_enet_ref$precision
res_enet_ref$recall
res_enet_ref$confusion

#Validamos el mejor cutoff

cuts <- seq(0.10, 0.90, by = 0.05)
resultados_enet_ref <- data.frame()

for (c in cuts) {
  met <- f1_eval(enet_prob_ref, valid_num_ref$Pobre, cutoff = c)
  
  resultados_enet_ref <- rbind(
    resultados_enet_ref,
    data.frame(
      cutoff = c,
      precision = as.numeric(met$precision),
      recall = as.numeric(met$recall),
      f1 = as.numeric(met$f1)
    )
  )
}

resultados_enet_ref
resultados_enet_ref[which.max(resultados_enet_ref$f1), ]


mejor_cutoff_enet_ref <- resultados_enet_ref$cutoff[which.max(resultados_enet_ref$f1)]
mejor_cutoff_enet_ref

#Predecimos  y validamos que no haya nnigun dato faltante para generar nuestra base 

test_prob_enet_ref <- predict(enet_fit_ref, newdata = test_num_ref, type = "prob")[, "Yes"]

test_pred_enet_ref <- ifelse(test_prob_enet_ref >= mejor_cutoff_enet_ref, 1, 0)
test_pred_enet_ref[is.na(test_pred_enet_ref)] <- 0
test_pred_enet_ref <- as.integer(test_pred_enet_ref)

submission_enet_ref <- data.frame(
  id = base_modelo_test$id,
  Pobre = test_pred_enet_ref
)

head(submission_enet_ref)
dim(submission_enet_ref)
sum(is.na(submission_enet_ref$Pobre))
unique(submission_enet_ref$Pobre)
names(submission_enet_ref)

write.csv(submission_enet_ref, "submission_model2-1_v1.csv",
          row.names = FALSE, quote = FALSE)
