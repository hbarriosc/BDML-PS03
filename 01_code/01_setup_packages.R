# Instalación de la librería Pacman
if (!requireNamespace("pacman", quietly = TRUE)) {
  install.packages("pacman")
}

# Agregamos la librerias necesarias
library(pacman)
p_load(rio, tidyverse, skimr, stargazer, rvest, dplyr, boot, knitr, kableExtra,
       janitor,ggplot2)
