library(tidyverse)
library(ggplot2)
library(patchwork)
library(plotrix)

setwd("/Users/Emma/Documents/GitHub/seeing-sums")
response.data <- read.table("30_ref_test_results.txt", header=TRUE)
ma.model.response.data <- read.table("30_ref_2_test_results.txt", header=TRUE)
setwd("/Users/Emma/Documents/GitHub/seeing-sums/Stimuli")
stimulus.data <- read.table("final_trial_info.txt", header=TRUE)

ma.response.data.tidy <- ma.model.response.data %>% 
  pivot_wider(names_from = Stimulus, values_from = ModelResponseMA)
names(ma.response.data.tidy) <- c("TrialNumber", "FirstImageArea", "SecondImageArea")
ma.model.response.data <- ma.response.data.tidy %>%
  mutate(ModelChoice = if_else(FirstImageArea > SecondImageArea, 1, 2))
test.results.ma.model <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
  merge(select(ma.model.response.data, TrialNumber, ModelChoice), by = "TrialNumber") %>% 
  mutate(More = if_else(ModelChoice==2, TRUE, FALSE)) %>% 
  mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>% 
  mutate(Ratio = if_else(Condition == "AA-Control", MARatio, if_else(Condition == "MA-Control", AARatio, 1.0))) %>% 
  select(-AARatio, -MARatio)

response.data.tidy <- response.data %>%
  pivot_wider(names_from = Stimulus, values_from = c(ModelResponseMA, ModelResponseAA))
# ma.model.response.data <- select(response.data.tidy, Trial, ModelResponseMA_1, ModelResponseMA_2)
aa.model.response.data <- select(response.data.tidy, Trial, ModelResponseAA_1, ModelResponseAA_2)
# names(ma.model.response.data) <- c("TrialNumber", "FirstImageMA", "SecondImageMA")
names(aa.model.response.data) <- c("TrialNumber", "FirstImageAA", "SecondImageAA")
# ma.model.response.data <- ma.model.response.data %>%
#   mutate(ModelChoice = if_else(FirstImageMA > SecondImageMA, 1, 2))
aa.model.response.data <- aa.model.response.data %>%
  mutate(ModelChoice = if_else(FirstImageAA > SecondImageAA, 1, 2))

test.results.ma.model <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
  merge(select(ma.model.response.data, TrialNumber, ModelChoice), by = "TrialNumber") %>% 
  mutate(Correct = if_else(ModelChoice==2, TRUE, FALSE)) %>% 
  mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>% 
  mutate(Ratio = if_else(Condition == "AA-Control", MARatio, if_else(Condition == "MA-Control", AARatio, 1.0))) %>% 
  select(-AARatio, -MARatio)
test.results.aa.model <- select(stimulus.data, TrialNumber, AARatio, MARatio, comparisonNumerosity) %>% 
  merge(select(aa.model.response.data, TrialNumber, ModelChoice), by = "TrialNumber") %>% 
  mutate(Correct = if_else(ModelChoice==2, TRUE, FALSE)) %>% 
  mutate(Condition = if_else(AARatio >= MARatio, if_else(AARatio == MARatio, "Control", "MA-Control"), "AA-Control")) %>% 
  mutate(Ratio = if_else(Condition == "AA-Control", MARatio, if_else(Condition == "MA-Control", AARatio, 1.0))) %>% 
  select(-AARatio, -MARatio)

# Graph MA Model Results Alone
results.graph.data.ma.model <- test.results.ma.model %>% 
  group_by(Condition, Ratio) %>% 
  summarize(Accuracy = mean(Correct), se = std.error(Correct))

aa.plot.ma.model <- ggplot(filter(results.graph.data.ma.model, Condition == "MA-Control"), aes(Ratio, Accuracy)) + 
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

ma.plot.ma.model <- ggplot(filter(results.graph.data.ma.model, Condition == "AA-Control"), aes(Ratio, Accuracy)) + 
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

control.plot.ma.model <- ggplot(filter(results.graph.data.ma.model, Condition == "Control"), aes (Ratio, Accuracy)) + 
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
  coord_equal(ratio = 3)

plot.ma.model <- (aa.plot.ma.model + control.plot.ma.model + ma.plot.ma.model)
plot.ma.model

# Graph AA Model Results Alone
results.graph.data.aa.model <- test.results.aa.model %>% 
  group_by(Condition, Ratio) %>% 
  summarize(Accuracy = mean(Correct), se = std.error(Correct))

aa.plot.aa.model <- ggplot(filter(results.graph.data.aa.model, Condition == "MA-Control"), aes(Ratio, Accuracy)) + 
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

ma.plot.aa.model <- ggplot(filter(results.graph.data.aa.model, Condition == "AA-Control"), aes(Ratio, Accuracy)) + 
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

control.plot.aa.model <- ggplot(filter(results.graph.data.aa.model, Condition == "Control"), aes (Ratio, Accuracy)) + 
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
  coord_equal(ratio = 3)

plot.aa.model <- (aa.plot.aa.model + control.plot.aa.model + ma.plot.aa.model)
plot.aa.model