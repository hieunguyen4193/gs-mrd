gc()
rm(list = ls())

library(dplyr)
library(tidyverse)
library(ggplot2)
library(ggpubr)
library(comprehenr)
library(scales)
library(stringr)

maindir <- "/media/hieunguyen/HNSD01/src/gs-mrd/model_files/10062024/features"

all.files <- Sys.glob(file.path(maindir, "*.csv"))

names(all.files) <- to_vec(
  for (item in all.files){
    str_split(str_replace(basename(item), ".csv", ""), "_")[[1]][[1]]
  }
)
path.to.main.src <- "/media/hieunguyen/HNSD01/src/gs-mrd"
meta.data <- read.csv(file.path(path.to.main.src, "final_metadata.csv"))
meta.data <- subset(meta.data, meta.data$True.label %in% c("+", "-")) %>%
  rowwise() %>%
  mutate(Label = ifelse(True.label == "+", "Cancer", "Control")) %>%
  mutate(input.SampleID = SampleID) %>% subset(select = -c(SampleID))

xlab.names <- list(
    EM = "All 256 4-mer end motifs",
    FLEN = "Fragment length",
    NUCLEOSOME = "Distance to nucleosome"
  )

feature.plot <- list()
all.featdf.pivot <- list()

for (feat in c("EM", "FLEN", "NUCLEOSOME")){
  print(sprintf("Working on %s", feat))
  featdf <- read.csv(all.files[[feat]])
  if (feat == "EM"){
    featdf.pivot <- featdf %>% subset(SampleID %in% meta.data$input.SampleID) %>% 
      pivot_longer(!SampleID, names_to = "feat", values_to = "val")  %>% rowwise() %>%
      mutate(Label = subset(meta.data, meta.data$input.SampleID == SampleID)$Label) %>%
      arrange(desc(Label))
    all.featdf.pivot[[feat]] <- featdf.pivot
    feature.plot[[feat]] <- featdf.pivot %>% 
      ggplot(aes(x = feat, y = val, fill = Label)) + 
      theme_pubr() +
      geom_bar(stat = "identity") + 
      theme(axis.text.x = element_blank(),
            axis.text.y = element_text(size = 25),
            axis.title = element_text(size = 25),
            legend.text = element_text(size = 25),
            legend.title = element_text(size = 25)) + 
      xlab(xlab.names[[feat]]) + ylab("Density") +
      scale_fill_manual(values = c("red", "gray")) +
      scale_x_discrete(breaks = unique(featdf$feat)[c(T, rep(F, 25))]) 
  } else {
    if (feat == "FLEN"){
      colnames(featdf) <- c(c("SampleID"), seq(50, 350))      
    } else {
      colnames(featdf) <- c(c("SampleID"), seq(-300, 300))
    }
    featdf.pivot <- featdf %>% subset(SampleID %in% meta.data$input.SampleID) %>% 
      pivot_longer(!SampleID, names_to = "feat", values_to = "val")  %>% rowwise() %>%
      mutate(Label = subset(meta.data, meta.data$input.SampleID == SampleID)$Label) %>%
      arrange(desc(Label))
    featdf.pivot$feat <- as.numeric(featdf.pivot$feat)
    
    all.featdf.pivot[[feat]] <- featdf.pivot
    feature.plot[[feat]] <- featdf.pivot %>% 
      ggplot(aes(x = feat, y = val, color = Label)) + 
      theme_pubr() +
      geom_line() + 
      theme(axis.text.x = element_blank(),
            axis.text.y = element_text(size = 25),
            axis.title = element_text(size = 25),
            legend.text = element_text(size = 25),
            legend.title = element_text(size = 25)) + 
      xlab(xlab.names[[feat]]) + ylab("Density") +
      scale_color_manual(values = c("red", "gray")) 
  }
}

control.sample <- "3-ZMC005B"
cancer.sample <- "6-HAGAAA11"

tmpdf <- all.featdf.pivot$FLEN %>%
  subset(SampleID %in% c(control.sample, cancer.sample)) %>%
  rowwise() %>%
  mutate(Label = ifelse(Label == "Cancer", "Input sample", "Healthy Ref."))

tmpdf %>% 
  ggplot(aes(x = feat, y = val, color = Label)) + 
  theme_pubr() +
  geom_line() + 
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 25),
        axis.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        legend.title = element_text(size = 25)) + 
  xlab(xlab.names[["FLEN"]]) + ylab("Density") +
  scale_color_manual(values = c("gray", "red")) 
  




control.sample <- "3-ZMC005B"
cancer.sample <- "7-ZMC057"

tmpdf <- all.featdf.pivot$EM %>%
  subset(SampleID %in% c(control.sample, cancer.sample)) %>%
  rowwise() %>%
  mutate(Label = ifelse(Label == "Cancer", "Input sample", "Healthy Ref."))

tmpdf %>% 
  ggplot(aes(x = feat, y = val, fill = Label)) + 
  theme_pubr() +
  geom_bar(stat = "identity") + 
  theme(axis.text.x = element_blank(),
        axis.text.y = element_text(size = 25),
        axis.title = element_text(size = 25),
        legend.text = element_text(size = 25),
        legend.title = element_text(size = 25)) + 
  xlab(xlab.names[["FLEN"]]) + ylab("Density") +
  scale_fill_manual(values = c("red", "gray")) 


