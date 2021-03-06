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

response.data.150 <- read.table("150_ref_test_results.txt", header=TRUE)
response.data.250 <- read.table("250_ref_test_results.txt", header=TRUE)

setwd("/Users/Emma/Documents/GitHub/seeing-sums/Stimuli")
stimulus.data <- read.table("final_trial_info.txt", header=TRUE)

process.results <- function(response.data){
  response.data.tidy <- response.data %>% 
    pivot_wider(names_from = Stimulus, values_from = ModelResponseMA)
  names(response.data.tidy) <- c("TrialNumber", "FirstImageArea", "SecondImageArea")
  model.response.data <- response.data.tidy %>%
    mutate(ModelChoice = if_else(FirstImageArea < SecondImageArea, 2, 1))
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
    geom_col(fill = "#008b09") + 
    scale_x_reverse() + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL)), limits = c(0, 1.0)) + 
    geom_errorbar(aes(ymin = Accuracy - se, ymax = Accuracy + se), width = 0.01) + 
    geom_hline(yintercept=0.5, linetype="dashed", color="black") + 
    labs(y = "Propensity to Choose More", x = "Additive Area") + 
    theme(axis.line = element_line(),
          axis.text = element_text(face = "bold"),
          axis.title.x = element_text(face = "bold", color = "#008b09"),
          axis.title = element_text(face = "bold"),
          panel.grid = element_blank(), 
          panel.background = element_blank())
  
  ma.plot <- ggplot(filter(results.graph.data, Condition == "AA-Control"), aes(Ratio, Accuracy)) + 
    geom_col(fill = "#df0101") + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL)), limits = c(0, 1.0)) + 
    geom_errorbar(aes(ymin = Accuracy - se, ymax = Accuracy + se), width = 0.01) +
    geom_hline(yintercept=0.5, linetype="dashed", color="black") + 
    labs(y = NULL, x = "Mathematical Area") + 
    theme(axis.text.y = element_blank(), 
          axis.ticks.y = element_blank(), 
          axis.line.x = element_line(),
          axis.text.x = element_text(face = "bold"),
          axis.title.x = element_text(face = "bold", color = "#df0101"),
          panel.grid = element_blank(), 
          panel.background = element_blank())
  
  plot <- (aa.plot + ma.plot)
  return(plot)
}

process.control.results <- function(response.data, nodes){
  response.data.tidy <- response.data %>% 
    pivot_wider(names_from = Stimulus, values_from = ModelResponseMA)
  names(response.data.tidy) <- c("TrialNumber", "FirstImageArea", "SecondImageArea")
  test.results <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
    merge(select(response.data.tidy, TrialNumber, FirstImageArea, SecondImageArea), by = "TrialNumber") %>% 
    mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>%
    filter(Condition == "Control") %>%
    mutate(Response = if_else(FirstImageArea == SecondImageArea, "Equal", if_else(FirstImageArea > SecondImageArea, "First", "Second"))) %>%
    select(-AARatio, -MARatio) %>%
    mutate(Nodes = nodes)
  return(test.results)
}

make.control.plot <- function(test.results){
  results.graph.data <- test.results %>%
    group_by(Response) %>%
    summarize(Frequency = n())
  control.plot <- ggplot(results.graph.data, aes(Response, Frequency, fill=Response)) + 
    geom_col() + 
    scale_y_continuous(expand = expand_scale(mult = c(0, NULL))) + 
    scale_fill_manual(limits=c("Equal", "First", "Second"), values = c("grey", "#016cff", "#466a9c")) + 
    labs(y = "Frequency", x = "Response") + 
    theme(axis.text.y = element_text(face = "bold", color = "black"),
          axis.ticks.y = element_line(color = "black"),
          axis.ticks.x = element_line(color = "black"),
          axis.text.x = element_text(face = "bold", color = "black"),
          axis.title = element_text(face = "bold", color = "black"),
          axis.line = element_line(color = "black"),
          panel.grid = element_blank(), 
          panel.background = element_blank())
}

test.results.20 <- process.results(response.data.20)
control.results.20 <- process.control.results(response.data.20, 20)
plot.20 <- make.plot(test.results.20)
control.plot.20 <- make.control.plot(control.results.20)

test.results.25 <- process.results(response.data.25)
control.results.25 <- process.control.results(response.data.25, 25)
plot.25 <- make.plot(test.results.25)
control.plot.25 <- make.control.plot(control.results.25)

test.results.30 <- process.results(response.data.30)
control.results.30 <- process.control.results(response.data.30, 30)
plot.30 <- make.plot(test.results.30)
control.plot.30 <- make.control.plot(control.results.30)

test.results.35 <- process.results(response.data.35)
control.results.35 <- process.control.results(response.data.35, 35)
plot.35 <- make.plot(test.results.35)
control.plot.35 <- make.control.plot(control.results.35)

test.results.40 <- process.results(response.data.40)
control.results.40 <- process.control.results(response.data.40, 40)
plot.40 <- make.plot(test.results.40)
control.plot.40 <- make.control.plot(control.results.40)

test.results.45 <- process.results(response.data.45)
control.results.45 <- process.control.results(response.data.45, 45)
plot.45 <- make.plot(test.results.45)
control.plot.45 <- make.control.plot(control.results.45)

test.results.150 <- process.results(response.data.150)
control.results.150 <- process.control.results(response.data.150, 150)
plot.150 <- make.plot(test.results.150)
control.plot.150 <- make.control.plot(control.results.150)

test.results.250 <- process.results(response.data.250)
control.results.250 <- process.control.results(response.data.250, 250)
plot.250 <- make.plot(test.results.250)
control.plot.250 <- make.control.plot(control.results.250)

# Numerosity analysis for control task results
all.control.data <- rbind(control.results.20, control.results.25, 
                          control.results.30, control.results.35, 
                          control.results.40, control.results.45) %>%
  select(-FirstImageArea, -SecondImageArea, -TrialNumber, -Condition) %>%
  mutate(Correct = if_else(Response=="Equal", TRUE, FALSE))
test.data <- all.control.data %>%
  group_by(comparisonNumerosity, Nodes) %>%
  summarize(Accuracy = mean(Correct))