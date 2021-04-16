## Twitter Sentiment with NLP
By Eduardo Osorio

## Objective:
The purpose of this report is to create algorithm that accurately predicts consumer sentiment from tweets. The algorithm is using tokenized version of individual tweets to predict the outcome in the form of positive or negative sentiment.

## The Data:
The data used today is from CrowdFlower. They manually sorted the data and classified it as positive, negative, no emotion, and I don't know. For the our purpose, I only used the positive and negative classes.

## Sources Used:
- https://data.world/crowdflower/brands-and-product-emotions
- https://medium.com/@acrosson/extract-subject-matter-of-documents-using-nlp-e284c1c61824
- https://machinelearningmastery.com/sparse-matrices-for-machine-learning/

## Summery of the Results:
- Consumers didn’t respond so favorably to google’s social media app feature called “Circles”.
- The model suggests that consumers didn’t like google+’s “Circles” because of its design.
- Consumers responded favorably towards the new apple products, Specifically the “IPad2”.
- People seemed to like products with google’s “Android operating system.

## Preliminary EDA:
I ran a basic Random Forest model to see a preview of the important features. The results were:
- 'sxsw', 7608
- 'mention', 5703
- 'rt', 2331
- 'google', 2059
- 'ipad', 1948
- 'apple', 1839
- 'quot', 1322
- 'iphone', 1230
- 'store', 1209
- "'s", 988

## Basemodel:
For this step I tokenized the data and lemmatized to reduce dimensionality. I ran both Random Forest and Naive Bayes models to see which one would be more accurate. They both did almost the same with an average testing accuracy of 65.5%.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Prelim%20RF%20Accuracy.png?token=APSW5OCGAKG3GCCAYQPPUDDAI62BM)


![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Prelim%20NB%20Accuracy.png?token=APSW5OBROL7WS6VRCSXYXGTAI62D6)

## Final Model:
For the final model I cut everything that was't classified as positive or negative. This cut the dataset by about 60% but increased the model accuracy over all. I also used TF-IDf to vectorize the tokenized tweets and assign weight to the more important features. This helped increase the average accuracy for both Random Forest and Naive Bayes by about 20%.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Final%20RF%20Accuracy.png?token=APSW5OCZPHCVZ6UYLC4PELDAI62HS)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Confusion%20matrix%20RF.png?token=APSW5OGUFNGR5ELYYFL3ARTAI62QG)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Final%20NB%20Accuracy.png?token=APSW5OA2UUKNS6QK2DAQQJDAI62KC)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Confusion%20matrix%20NB.png?token=APSW5ODRGBF5HJI3NFOWJFDAI62SW)

## Post EDA:
The most common negative and positive words are very similar with a few words that stand out. Out of the top ten most common positive words:
- Great
- Party
- Launch

Stand out the most since they're not in the top ten most common negative words.

Out of the top ten most negative words:

- Design
- Social
- Circle
- Need

Stand out the most since they're not in the top ten most common positive words.

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Most%20pos%20sentiments.png?token=APSW5OHNEYMCH736PZRQ2HTAI63US)

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Most%20neg%20sentiments.png?token=APSW5OEPXCMEEG42Q3EYBY3AI63V2)

Here is an example of a tweet the model thought was positive:

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Pos%20Tweet%20example.png?token=APSW5OAU5PEBACIACEQQVLDAI6356)

Here is an example of a tweet the model though was negative:

![alt text](https://raw.githubusercontent.com/Eduardoosorio23/Mod_4_Project/main/Data/Pictures/Neg%20Tweet%20example.png?token=APSW5ODRZTREQGY3HDPFDQ3AI637G)

## Recommendations:
- If google wishes to work on it’s social media apps, they should really pay attention to the design and functionality.
- Apple should just keep releasing new products in order to keep its fanbase excited
