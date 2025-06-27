#!/usr/bin/env python3

'''
    "Meta labels" in the context of testing a transformer model typically refer to
    higher-level annotations or metadata** associated with the dataset or evaluation process. 
    Their meaning depends on the specific use case:

    ### Possible Interpretations:
    1. **Dataset-Level Labels**:  
       - These could be labels indicating categories of data (e.g., "formal", "informal", }
       "scientific") that help in analyzing model performance across different subsets.
       
    2. **Evaluation Meta Labels**:  
       - Labels used to define the type of evaluation (e.g., "correct", "incorrect", 
       "partially correct") in a custom testing framework.
       
    3. **Self-Supervised or Weakly-Supervised Learning**:  
       - In cases where transformers are trained with self-supervised objectives, 
       meta labels might indicate artificially generated supervision signals.

    4. **Explainability & Interpretability Labels**:  
       - Meta labels may represent additional data used for explainability 
       (e.g., attention distributions, confidence scores).

    5. **Hierarchical Labeling**:  
       - If a model deals with multiple levels of labels (e.g., sentiment classification 
       with both fine-grained and coarse-grained categories), the higher-level ones can be 
       considered meta labels.


   source venv/bin/activate 

'''


from get_tokens import get_top_ten_tokens
from transformers import pipeline
import csv
import numpy as np

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


mails = []

with open('spam.csv', 'r') as file:
   reader = csv.reader(file)
   for row in reader:
      mails.append(row)


diseases = {
   
   "mail" : ['spam', 'ham']

}


i = 0

best_scores = []

best_scores_labels = []

for mail in mails:

   print(f"No. {i+1}")
   

   for j in diseases.keys():
      
      result = classifier(mail[1], diseases.get(j))
      best_scores.append(max(result.get('scores')))
      best_scores_labels.append(result.get('labels')[np.argmax(result.get('scores'))])
      #print(result)

   print(f"The diagnosed is {best_scores_labels[np.argmax(best_scores)]}, with a probability of {max(best_scores)} !!")

   with open('diagnostic.csv', 'a', newline='', encoding='utf-8') as csvfile:
      writer = csv.writer(csvfile) 
      writer.writerow([best_scores_labels, best_scores]) #labeled by Transformer 
      writer.writerow([mail[0]]) # original label

      es_correcta_clasificacion_transformador = lambda x, y: "C" if x == y else "I"

      print(f"Classification?: {es_correcta_clasificacion_transformador(best_scores_labels, [mail[0]])}")

      writer.writerow(es_correcta_clasificacion_transformador(best_scores_labels, [mail[0]])) #lambda

      try:

         writer.writerow(get_top_ten_tokens(mail[1]) + [mail[0]])

      except:

         writer.writerow(["ERROR"])


   print("\n=================================================================================================================")

   best_scores = []
   best_scores_labels = []
   i+=1





   