library(ggplot2)

# Before reduced

# Load raw data
train.1 <- read.csv("4_emo.csv", header = TRUE)

train$Utterances <- as.character(train$Utterances)
train$Label <- as.factor(train$Label)
str(train)
ggplot(train.1, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")

# After reduced

# Load raw data
train.2 <- read.csv("4_emo_reduced.csv", header = TRUE)

train$Utterances <- as.character(train$Utterances)
train$Label <- as.factor(train$Label)
str(train)
ggplot(train.2, aes(x = Utterances, fill = Label)) +
  geom_bar(width = 1) +
  xlab("Utterance") +
  ylab("Label") +
  labs(fill = "Label") +
  ggtitle("Label distribution")
  