#==============================================================================#
# 1. LIBRERÍAS
library(tidyverse)
library(caret)
library(pROC)
library(patchwork)   
library(PRROC)

#==============================================================================#
# 2. CARGA DE DATOS
test_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_hogares.csv")
test_personas <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/test_personas.csv")
train_hogares <- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_hogares.csv")
train_personas<- read.csv("C:/Users/Energy/Desktop/Personal/MECA/4to Semestre/BDML/Taller 2/Data/train_personas.csv")

#==============================================================================#
# 3. WRANGLING

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

#==============================================================================#
# 4. ENTRENAMIENTO DEL MODELO LOGIT

set.seed(123)
# Creamos el set sin id
train_logit_input <- train_imp %>% select(-id)

control_logit <- trainControl(
  method = "cv", 
  number = 5, 
  classProbs = TRUE, 
  summaryFunction = twoClassSummary
)

modelo_logit <- train(
  Pobre ~ ., 
  data = train_logit_input,  # <--- CAMBIO NECESARIO PARA ID
  method = "glm", 
  family = "binomial",
  trControl = control_logit
)

#==============================================================================#
# 5 GRÁFICAS

# Preparar datos de validación interna
pred_probs <- predict(modelo_logit, train_logit_input, type = "prob")
eval_df <- data.frame(
  obs = train_imp$Pobre,
  Si = pred_probs$Si,
  No = pred_probs$No
)

#-----------------------------------------------------------
# 5.1. CURVA PRECISION-RECALL (PR)

# Es mejor que la ROC cuando las clases están desbalanceadas.
pr_obj <- pr.curve(scores.class0 = eval_df$Si[eval_df$obs == "Si"],
                   scores.class1 = eval_df$Si[eval_df$obs == "No"], 
                   curve = TRUE)

plot(pr_obj, main = "1. Curva Precision-Recall", color = "#8e44ad")

#-----------------------------------------------------------
# 5.2. GRÁFICO DE CALIBRACIÓN

# Evalúa si una probabilidad del 80% realmente corresponde al 80% de los casos.
cal_obj <- calibration(obs ~ Si, data = eval_df, cuts = 10)

ggplot(cal_obj) + 
  geom_line(color = "#e67e22", size = 1) + 
  geom_point(size = 3) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  labs(title = "2. Curva de Calibración", 
       subtitle = "Cercanía a la diagonal = Probabilidades realistas",
       x = "Probabilidad Predicha", y = "Proporción Observada") +
  theme_minimal()

#-----------------------------------------------------------
# 5.3. CREAR Y GRAFICAR CURVA ROC

# 5.3.1 Cálculo de probabilidades (entrenamiento sin id)
pred_prob_logit <- predict(modelo_logit, train_logit_input, type = "prob")

# 5.3.2. Roc (Comparación entre valores reales -train_imp$Pobre- con la probabilidad de "Si"
roc_logit <- roc(train_imp$Pobre, pred_prob_logit$Si)

# 5.3.3. Grafica de roc_logit
plot(roc_logit, 
     col = "#27ae60", 
     lwd = 4, 
     main = "Curva ROC - Modelo Logit")

# 5.3.4. Adicionar valor del AUC
text(0.4, 0.2, paste("AUC =", round(auc(roc_logit), 4)), cex = 1.2, col = "darkgreen")

#==============================================================================#
# 6. PREDICCIÓN FINAL PARA TEST

test_logit_input <- test_imp %>% select(-id)

final_preds_logit <- predict(modelo_logit, newdata = test_logit_input)

submission_logit <- data.frame(
  id = test_imp$id, 
  Pobre = ifelse(final_preds_logit == "Si", 1, 0)
)

write.csv(submission_logit, "predicciones_pobreza_logit.csv", row.names = FALSE)