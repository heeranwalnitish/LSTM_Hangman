# Introduction

Hangman is a popular word game. Person 1 chooses a word at his will and the goal of Person 2 is to predict the word by predicting one character at a time. At every step, Person 1 gives the following feedback to Person 2 depending on the predicted character:
1. If the character is present, Person 1 informs Person 2 of all the positions at which the character is present.
e.g. if the word is 'hello' and Person 2 predicts 'e', Person 1 has to report that the character 'e' is present at position 2
2. If the character is not present, Person 1 simply reports that the character is not present in his/her word.

Refer to [this](https://en.wikipedia.org/wiki/Hangman_(game)) Wikipedia article for more details.
