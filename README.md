# Analyzing Media Bias using Deep Learning
The purpose of this project was to use deep learning to implement a text classifier that classifies liberal and conservative biases in news article headings. An LSTM (Long Short Term Memory) neural network was implemented to capture the biases. The [All The News dataset](https://www.kaggle.com/snapcrack/all-the-news), consisting of 143,000 articles was used to train and test the deep learning model. The model was run on another held out dataset containing manually downloaded article headings and the results were visualized for analysis using Tableau. This repository also includes a  presentation of the project as well as a paper.

## Files Included:

| File    | Description    | 
| ------------- |:-------------:| 
| lstm.py   |The main script for the LSTM model.|
|Media Bias_Final.twbx|Tableau file containing data visualizations and analysis results.|
|Analyzing Media Bias.pptx| A brief presentation of the project outline and results.|
|Media Analysis Final Paper.pdf| A paper detailing the project description, literature reviews conducted, methods, and ways to improve and extend on the project.|

## Instructions to Use:
1. Download data from [here](https://www.kaggle.com/snapcrack/all-the-news).
2. Put the three csv files downloaded (articles1.csv, articles2.csv, articles3.csv) in a folder and name it 'data'.
3. Make sure the python script 'lstm.py' is in the same directory as 'data'.
4. Run the python script 'lstm.py'
