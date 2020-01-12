library(tidyverse)
library(ggplot2)

setwd("/Users/Emma/Documents/GitHub/seeing-sums")
distance.data <- read.table("test_representation_distance_paired.txt", header=TRUE)
names(distance.data) <- c("TrialNumber", "Distance")

setwd("/Users/Emma/Documents/GitHub/seeing-sums/Stimuli")
stimulus.data <- read.table("final_trial_info.txt", header=TRUE)

test.results <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
  merge(select(distance.data, TrialNumber, Distance), by = "TrialNumber") %>% 
  mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>% 
  mutate(Ratio = if_else(Condition == "AA-Control", MARatio, if_else(Condition == "MA-Control", AARatio, 1.0))) %>% 
  select(-AARatio, -MARatio)

aggregate.results <- test.results %>%
  group_by(Condition, Ratio) %>%
  summarize(MeanEuclideanDistance = mean(Distance))

aggregate.results.collapsed <- test.results %>%
  group_by(Condition) %>%
  summarize(MeanEuclideanDistance = mean(Distance))

# ANOVA
fit <- aov(Distance ~ Condition + Ratio + Condition:Ratio, data=test.results)
summary(fit)