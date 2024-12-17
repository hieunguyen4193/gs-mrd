gc()
rm(list = ls())

library(ggpubr)
library(tidyverse)
library(pROC)
library(ggplot2)
library(dplyr)
library(stringr)

maindir <- "/media/hieunguyen/HNSD01/src/gs-mrd/model_files/10062024"
path.to.save.figures <- file.path(maindir, "figures")
dir.create(path.to.save.figures, showWarnings = FALSE, recursive = TRUE)

meta.data <- read.csv(file.path(maindir, "metadata.csv")) %>%
  rowwise() %>%
  mutate(SampleID2 = str_split(SampleID, "-")[[1]][[2]])

pos.samples <- subset(meta.data, meta.data$True.label == "+")$SampleID2
neg.samples <- subset(meta.data, meta.data$True.label == "-")$SampleID2

set.seed(411)
flendf <- read.csv(file.path(maindir, "features", "FLEN_features.csv")) %>%
  rowwise() %>%
  mutate(SampleID = str_split(SampleID, "-")[[1]][[2]]) %>%
  subset(SampleID %in% c(sample(pos.samples, 20), sample(neg.samples, 20))) %>%
  column_to_rownames("SampleID") %>% t() %>% as.data.frame()

flendf$size <- seq(50, 350)

flendf.pivot <- flendf %>% pivot_longer(!size, names_to = "SampleID", values_to = "freq")
flendf.pivot <- merge(flendf.pivot, subset(meta.data, select = c(SampleID2, True.label)),
                      by.x = "SampleID", by.y= "SampleID2")
colnames(flendf.pivot) <- c("SampleID", "size", "freq", "Label")

flen.plot <- flendf.pivot %>% ggplot(aes(x = size, y = freq, color = Label)) + 
  geom_line(alpha = 0.5) + 
  theme_pubr() + 
  scale_color_manual(values = c("black", "red")) + 
  theme(legend.title = element_text(size = 25),
        axis.text = element_text(size = 25),
        axis.title = element_text(size = 25),
        legend.position = "right",
        legend.text = element_text(size = 25)) +
  xlab("Fragment size") + ylab("Frequency")
  
ggsave(plot = flen.plot, filename = "FLEN.svg", path = path.to.save.figures, device = "svg", dpi = 300, width = 14, height = 10)
ggsave(plot = flen.plot, filename = "FLEN.png", path = path.to.save.figures, device = "png", dpi = 300, width = 14, height = 10)

