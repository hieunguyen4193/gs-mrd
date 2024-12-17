gc()
rm(list = ls())

library(ggpubr)
library(tidyverse)
library(pROC)
library(ggplot2)
library(dplyr)
library(stringr)
library(scales)
maindir <- "/media/hieunguyen/HNSD01/src/gs-mrd/model_files/10062024"
path.to.save.figures <- file.path(maindir, "figures")
dir.create(path.to.save.figures, showWarnings = FALSE, recursive = TRUE)

meta.data <- read.csv(file.path(maindir, "metadata.csv")) %>%
  rowwise() %>%
  mutate(SampleID2 = str_split(SampleID, "-")[[1]][[2]])

pos.samples <- subset(meta.data, meta.data$True.label == "+")$SampleID2
neg.samples <- subset(meta.data, meta.data$True.label == "-")$SampleID2

maindf <- read.csv(file.path(maindir, "traindf.csv")) %>%
  subset(True.label != "?") %>%
  rowwise() %>%
  mutate(Label = ifelse(True.label == "+", 1, 0))

all.features <- setdiff(colnames(maindf), c("SampleID", "RUN", "Group_RUN", "Cancer", "True.label", "True_label", "Label"))  
rocobj <- list()

for (feat in all.features){
  tmp <- roc(maindf$Label, maindf[[feat]])
  auc.val <- as.numeric(auc(tmp))
  feat <- sprintf("%s (AUC = %s)", feat, round(auc.val, 2))
  rocobj[[feat]] <- tmp  
}

auc.plot <- ggroc(rocobj, size = 1, legacy.axes = TRUE) +
  theme_pubr() +
  theme(axis.text.x = element_text(size = 25, angle = 0, margin = margin(t = 15)),
        axis.text.y = element_text(size = 25, angle = 0),
        axis.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        legend.title = element_text(size = 25),
        legend.position = "bottom") + 
  guides(color=guide_legend(title="Model", ncol = 2, nrow = 6)) +
  xlab("1 - Specificity") + ylab("Sensitivity") + 
  scale_color_manual(values = hue_pal()(length(rocobj)))

ggsave(plot = auc.plot, filename = "AUC_full_388_samples.svg", path = path.to.save.figures, device = "svg", dpi = 300, width = 16, height = 10)
ggsave(plot = auc.plot, filename = "AUC_full_388_samples.png", path = path.to.save.figures, device = "png", dpi = 300, width = 16, height = 10)

##### boxplots
dir.create(file.path(path.to.save.figures, "boxplots"), showWarnings = FALSE, recursive = TRUE)
for (feat in all.features){
  bp <- maindf %>%
    rowwise() %>%
    mutate(Label = ifelse(Cancer == "Healthy", "Healthy", True.label)) %>% 
    ggplot(aes_string(x = "Label", y = feat)) + geom_boxplot() +
    theme_pubr() +
    theme(axis.text.x = element_text(size = 25, angle = 0, margin = margin(t = 15)),
          axis.text.y = element_text(size = 25, angle = 0),
          axis.title = element_text(size = 25),
          legend.text = element_text(size = 25),
          legend.title = element_text(size = 25),
          legend.position = "bottom") +
    xlab("Label") + ylab(feat) +
    stat_compare_means(comparisons = list(c("-", "+"), c("Healthy", "+")), method = "t.test", 
                       symnum.args = list(cutpoints = c(0, 0.0001, 0.001, 0.01, 0.05, Inf), symbols = c("****", "***", "**", "*", "ns")))
  ggsave(plot = bp, filename = sprintf("boxplot_%s.svg", feat), path = file.path(path.to.save.figures, "boxplots"), device = "svg", dpi = 300, width = 10, height = 10)
  ggsave(plot = bp, filename = sprintf("boxplot_%s.png", feat), path = file.path(path.to.save.figures, "boxplots"), device = "png", dpi = 300, width = 10, height = 10)
}
