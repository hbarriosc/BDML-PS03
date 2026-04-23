# Instalación de la librería Pacman
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

# Agregamos la librerias necesarias
library(pacman)
p_load(tidyverse,dplyr,caret,glmnet,rpart,e1071,pROC,patchwork,PRROC,ranger,ROSE,here)
