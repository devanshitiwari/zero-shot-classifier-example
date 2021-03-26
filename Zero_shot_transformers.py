
import ftfy
import string
from torch.optim.adam import Adam
import re, nltk, spacy
import pandas as pd
import numpy as np
#import language_tool_python
from transformers import pipeline
from numpy import argmax

#tool = language_tool_python.LanguageTool('en-US')
zero_shot_classifier = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')

#Start out with creating a hypothesis template. The Null Hypothesis is status quo and the alternate is that text refers to the new topic.
hypothesis_template = "This text is about {}."


#creating a dictionary of topics and the subtopics that belong in each layer. This can also be done by creating a matrix of layers.
thisdict = {
    
    "A":'"a1","a2","a3"',
    "B":'"b1","b2","b3"',
    "C":'"c1","c2","c3"'
}

#main topics in an array
classes = ['A','B','C']


d = []
predicted_class = ""
for i, row in df.iterrows():
    predicted_class = ""
    text = row['text']
    text = ftfy.fix_text(text)
    #fix punctuation
    text = text.translate(str.maketrans(dict.fromkeys(string.punctuation)))
    #fix grammar within each comment
    text = tool.correct(text)
    topic = row['topic']
    
    #using classifier to generate hypothesis statements and testing against the data.
    
    results = zero_shot_classifier(text, classes, hypothesis_template=hypothesis_template, multi_class=True)
    
    #assigning scores
    SCORES = results["scores"]
    #assigning labels
    CLASSES = results["labels"]
    
    #getting the highest score's index
    
    BEST_INDEX = argmax(SCORES)
    predicted_class = CLASSES[BEST_INDEX]
    # Print sentence with predicted labels
    print("iter: "+text+" TOPIC:"+predicted_class)
    
    #in this method We will be focusing on the top four topics and then explore the topmost subtopic within each. We can extend the logic as per the usecase.
    
    #for first topic
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    #assigning scores
    SCORES_sub = results_sub["scores"]
    #assigning labels
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub1 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
    #for second topic
    predicted_class = CLASSES[BEST_INDEX+1]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub2 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
    
    #for third topic
    predicted_class = CLASSES[BEST_INDEX+2]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub3 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
       
    #for fourth topic
    predicted_class = CLASSES[BEST_INDEX+3]
    nc = thisdict[predicted_class]
    results_sub = zero_shot_classifier(text, nc, hypothesis_template=hypothesis_template, multi_class=True)
   
    SCORES_sub = results_sub["scores"]
    CLASSES_sub = results_sub["labels"]
    BEST_INDEX_sub = argmax(SCORES_sub)
    predicted_class_sub4 = CLASSES_sub[BEST_INDEX_sub]
    #print(predicted_class_sub)
    #print("iter: "+text+" sub topic:" + str(predicted_class_sub))
    
    #append everything to the dataframe. 
        
    d.append(
            {
                'text': row['text'],
                'topic': row['topic'],
                'predicted_class1': CLASSES[BEST_INDEX],
                'predicted_class2': CLASSES[BEST_INDEX+1],
                'predicted_class3': CLASSES[BEST_INDEX+2],
                'predicted_class4': CLASSES[BEST_INDEX+3],
                'predicted_sub1': predicted_class_sub1,
                'predicted_sub2': predicted_class_sub2,
                'predicted_sub3': predicted_class_sub3,
                'predicted_sub4': predicted_class_sub4
            }
        )

    #output file write
    
    output = pd.DataFrame(d)
    output.to_csv('topic-model-input-classified.csv', index=False)


        

        
        
        
        
        
        
        
        
