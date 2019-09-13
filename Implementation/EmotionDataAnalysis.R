library(ggplot2)

# Before reduced
# Load raw data
train.1 <- read.csv("4_emo.csv", header = TRUE)
train.1$Utterances <- as.character(train.1$Utterances)
train.1$Label <- as.factor(train.1$Label)
str(train.1)
ggplot(train.1, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")

# After reduced
# Load raw data
train.2 <- read.csv("4_emo_reduced.csv", header = TRUE)
train.2$Utterances <- as.character(train.2$Utterances)
train.2$Label <- as.factor(train.2$Label)
str(train.2)
ggplot(train.2, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")

# 30k label data
# Load raw data
train.2 <- read.csv("starterkitdata/train.", header = TRUE)
train.2$Utterances <- as.character(train.2$Utterances)
train.2$Label <- as.factor(train.2$Label)
str(train.2)
ggplot(train.2, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")
