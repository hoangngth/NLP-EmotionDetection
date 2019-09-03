#Load raw data
train <- read.csv("4_emo.csv", header = TRUE)

library(ggplot2)

train$Utterances <- as.character(train$Utterances)
train$Label <- as.factor(train$Label)
str(train)
ggplot(train, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")
  