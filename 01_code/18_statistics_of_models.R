logit_prob2 <- predict(logit_fit2, newdata = valid_split2, type = "response")

# 1. Curva ROC usando validación
roc_logit2 <- roc(
  response = valid_split2$Pobre,
  predictor = logit_prob2,
  levels = c("No", "Yes"),
  direction = "<"
)

png(here("02_output","figures","roc_logit_mejorado.png"),
    width = 900, height = 700)

# 2. Graficar ROC
plot(
  roc_logit2,
  col = "#27ae60",
  lwd = 4,
  main = "Curva ROC - Logit mejorado"
)

# 3. Agregar AUC al gráfico
text(
  0.6, 0.2,
  paste("AUC =", round(auc(roc_logit2), 4)),
  cex = 1.2,
  col = "darkgreen"
)

dev.off()
# 4. Ver AUC en consola
auc(roc_logit2)

#2 GRAFICA

#Lo que realizaremos a contunuacion es un grafica que nos muestre algunas de las variables en 
#donde el modelo no solo predice, también muestra que  educación, empleo y protección social son dimensiones
#centrales de la pobreza.

# Importancia de variables del Logit mejorado
coef_logit2 <- summary(logit_fit2)$coefficients

importancia_logit2 <- data.frame(
  variable = rownames(coef_logit2),
  coeficiente = coef_logit2[, "Estimate"],
  p_value = coef_logit2[, "Pr(>|z|)"],
  stringsAsFactors = FALSE
)

importancia_logit2 <- importancia_logit2[importancia_logit2$variable != "(Intercept)", ]
importancia_logit2$importancia <- abs(importancia_logit2$coeficiente)

# Mirar nombres reales para validar
print(importancia_logit2$variable)

# Elegimos las variables que sí aparecen fuerte en tu gráfica original
vars_clave <- c(
  "prom_P6090",  # Afiliación salud
  "prom_P6240",  # Actividad
  "prom_P6430",  # Tipo empleo
  "prom_P6210"   # Educación
)

idx_vars <- unlist(lapply(vars_clave, function(x) {
  grep(x, importancia_logit2$variable)
}))

importancia_filtrada <- importancia_logit2[idx_vars, ]

# Nos quedamos con el coeficiente más alto de cada grupo
importancia_filtrada$grupo <- NA
importancia_filtrada$grupo[grepl("prom_P6090", importancia_filtrada$variable)] <- "Afiliación a salud"
importancia_filtrada$grupo[grepl("prom_P6240", importancia_filtrada$variable)] <- "Actividad laboral"
importancia_filtrada$grupo[grepl("prom_P6430", importancia_filtrada$variable)] <- "Tipo de empleo"
importancia_filtrada$grupo[grepl("prom_P6210", importancia_filtrada$variable)] <- "Educación"

importancia_final <- aggregate(
  importancia ~ grupo,
  data = importancia_filtrada,
  FUN = max
)

importancia_final <- importancia_final[order(importancia_final$importancia), ]

write.csv(
  importancia_final,
  here("02_output", "tables", "importancia_variables_logit.csv"),
  row.names = FALSE
)

# Gráfica
png(
  here("02_output", "figures", "factores_clave_pobreza.png"),
  width = 900,
  height = 700
)

par(mar = c(5, 10, 3, 2))

barplot(
  importancia_final$importancia,
  names.arg = importancia_final$grupo,
  horiz = TRUE,
  las = 1,
  col = "#2E86C1",
  border = NA,
  main = "Factores clave de la pobreza",
  cex.main = 1,
  cex.names = 0.9,
  xlab = "Importancia (|coeficiente|)"
)

dev.off()

par(mar = c(5, 4, 4, 2))

# Grafica 3

#Para esta grafica La subcobertura representa hogares pobres que quedarían por fuera del programa;
#la fuga representa hogares no pobres que recibirían beneficios.El mejor cutoff busca equilibrar ambos errores mediante F1 
#los falsos negativos = hogares pobres clasificados como no pobres = subcobertura
# falsos positivos = hogares no pobres clasificados como pobres = fuga.


pred_valid_logit2 <- ifelse(logit_prob2 >= mejor_cutoff_logit2, "Yes", "No")
pred_valid_logit2 <- factor(pred_valid_logit2, levels = c("No", "Yes"))

real_valid_logit2 <- factor(valid_split2$Pobre, levels = c("No", "Yes"))

cm_logit2 <- caret::confusionMatrix(
  pred_valid_logit2,
  real_valid_logit2,
  positive = "Yes"
)

cm_logit2$table

mat <- cm_logit2$table

TN <- mat["No", "No"]
FN <- mat["No", "Yes"]
FP <- mat["Yes", "No"]
TP <- mat["Yes", "Yes"]

tasa_subcobertura <- FN / (FN + TP)
tasa_fuga <- FP / (FP + TN)

errores_politica <- c(
  "Subcobertura\nFalsos negativos" = tasa_subcobertura,
  "Fuga\nFalsos positivos" = tasa_fuga
)

png(
  here("02_output", "figures", "errores_politica_logit.png"),
  width = 900,
  height = 700
)

barplot(
  errores_politica,
  ylim = c(0, max(errores_politica) + 0.1),
  main = "Errores de política pública - Logit",
  ylab = "Tasa de error",
  col = "#2E86C1",
  border = NA,
  las = 1
)

text(
  x = c(0.7, 1.9),
  y = errores_politica + 0.03,
  labels = paste0(round(errores_politica * 100, 1), "%")
)

dev.off()