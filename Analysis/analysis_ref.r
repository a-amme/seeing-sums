library(tidyverse)
library(ggplot2)
library(patchwork)
library(plotrix)

setwd("/Users/Emma/Documents/GitHub/seeing-sums")
response.data.20 <- read.table("20_ref_test_results.txt", header=TRUE)
response.data.25 <- read.table("25_ref_test_results.txt", header=TRUE)
response.data.30 <- read.table("30_ref_2_test_results.txt", header=TRUE)
response.data.35 <- read.table("35_ref_test_results.txt", header=TRUE)
response.data.40 <- read.table("40_ref_test_results.txt", header=TRUE)
response.data.45 <- read.table("45_ref_test_results.txt", header=TRUE)

setwd("/Users/Emma/Documents/GitHub/seeing-sums/Stimuli")
stimulus.data <- read.table("final_trial_info.txt", header=TRUE)

process.results <- function(response.data){
  response.data.tidy <- response.data %>% 
    pivot_wider(names_from = Stimulus, values_from = ModelResponseMA)
  names(response.data.tidy) <- c("TrialNumber", "FirstImageArea", "SecondImageArea")
  model.response.data <- response.data.tidy %>%
    mutate(ModelChoice = if_else(FirstImageArea > SecondImageArea, 1, 2))
  test.results <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
    merge(select(model.response.data, TrialNumber, ModelChoice), by = "TrialNumber") %>% 
    mutate(More = if_else(ModelChoice==2, TRUE, FALSE)) %>% 
    mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>% 
    mutate(Ratio = if_else(Condition == "AA-Control", MARatio, if_else(Condition == "MA-Control", AARatio, 1.0))) %>% 
    select(-AARatio, -MARatio)
  return(test.results)
}

make.plot <- function(test.results){
  results.graph.data <- test.results %>% 
    group_by(Condition, Ratio) %>% 
    summarize(Accuracy = mean(More), se = std.error(More))
  
  aa.plot <- ggplot(filter(results.graph.data, Condition == "MA-Control"), aes(Ratio, Accuracy)) + 
    geom_col(fill = "green") + 
    scale_x_reverse() + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL)), limits = c(0, 1.0)) + 
    geom_errorbar(aes(ymin = Accuracy - se, ymax = Accuracy + se), width = 0.01) + 
    labs(y = "Propensity to Choose More", x = "Additive Area") + 
    theme(axis.line = element_line(),
          axis.text = element_text(face = "bold"),
          axis.title.x = element_text(face = "bold", color = "green"),
          axis.title = element_text(face = "bold"),
          panel.grid = element_blank(), 
          panel.background = element_blank())
  
  ma.plot <- ggplot(filter(results.graph.data, Condition == "AA-Control"), aes(Ratio, Accuracy)) + 
    geom_col(fill = "red") + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL)), limits = c(0, 1.0)) + 
    geom_errorbar(aes(ymin = Accuracy - se, ymax = Accuracy + se), width = 0.01) + 
    labs(y = NULL, x = "Mathematical Area") + 
    theme(axis.text.y = element_blank(), 
          axis.ticks.y = element_blank(), 
          axis.line.x = element_line(),
          axis.text.x = element_text(face = "bold"),
          axis.title.x = element_text(face = "bold", color = "red"),
          panel.grid = element_blank(), 
          panel.background = element_blank())
  
  control.plot <- ggplot(filter(results.graph.data, Condition == "Control"), aes (Ratio, Accuracy)) + 
    geom_col(fill = "blue", width = 1) + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL)), limits = c(0, 1.0)) + 
    geom_errorbar(aes(ymin = Accuracy - se, ymax = Accuracy + se), width = 0.03) + 
    scale_x_continuous(breaks=c(1.0)) + 
    labs(y = NULL, x = "Control") + 
    theme(axis.text.y = element_blank(), 
          axis.ticks.y = element_blank(), 
          axis.line.x = element_line(),
          axis.text.x = element_text(face = "bold"),
          axis.title.x = element_text(face = "bold", color = "blue"),
          panel.grid = element_blank(), 
          panel.background = element_blank()) + 
    coord_fixed(ratio = 3)
  
  plot <- (aa.plot + control.plot + ma.plot)
  return(plot)
}

test.results.20 <- process.results(response.data.20)
plot.20 <- make.plot(test.results.20)

test.results.25 <- process.results(response.data.25)
plot.25 <- make.plot(test.results.25)

test.results.30 <- process.results(response.data.30)
plot.30 <- make.plot(test.results.30)

test.results.35 <- process.results(response.data.35)
plot.35 <- make.plot(test.results.35)

test.results.40 <- process.results(response.data.40)
plot.40 <- make.plot(test.results.40)

test.results.45 <- process.results(response.data.45)
plot.45 <- make.plot(test.results.45)