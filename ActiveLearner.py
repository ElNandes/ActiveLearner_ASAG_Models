from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from flair.data import Sentence
from flair.models import SequenceTagger
from scipy.optimize import linear_sum_assignment
from nltk.tokenize import sent_tokenize
import pandas as pd
import numpy as np
embedder_model = SentenceTransformer('all-MiniLM-L6-v2')
tagger = SequenceTagger.load("flair/chunk-english")




class ActiveLearner:
    
    def __init__(self):
        pass
    
    def get_uncertainty_samples(self,data,percentage,question_id):
        uncertain_samples_df = data.loc[(data['Answer ID']==question_id)]

        return uncertain_samples_df.nsmallest(int(len(uncertain_samples_df.index)*percentage),
                                              columns="Uncertainty Score")
    
    def compute_uncertainty_scores(self):
        pass
    
    def save_file(self):
        pass
    
    def stratergy_1(self):
        pass
    
    def stratergy_2(self):
        pass
    
    def stratergy_3(self):
        pass
    
    def create_learner(self,estimator = RandomForestClassifier(),query_strategy=uncertainty_sampling):
        return ActiveLearner(
            estimator = estimator,
            query_strategy = query_strategy)
    
    def select_add_feedback(self,feedback_file_name):
        feedbacks = pd.read_csv(feedback_file_name)['Feedbacks']
        print("What feedback would you give this answer?")
        for i,f in enumerate(feedbacks):
            print(i,":",f)
            
        feedback_option = input("\n0: Add from existing feedback.\n1: Add new feedback.")
        if feedback_option == '0':
            feedback_id = np.array([int(input("\nSelect feedback ID: "))], dtype=int)
        elif feedback_option == '1':
            feedback_statement = input("\nEnter new feedback: ")

            feedbacks.append(feedback_statement)
            feedback_id = np.array([feedbacks.index(feedback_statement)], dtype=int)
            print("feedback ID: ",feedback_id)
        else:
            print("\nWrong ID selected: ")
         
        return feedback_id
        #learner1.teach(student_answer_embedding.reshape(1,-1),feedback_id_s1)
    
    def generate_score(self):
        pass
    
    def store_facts(self,facts_data,columns,filename):
        facts_df = pd.DataFrame(data=facts_data,index=False)
        facts_df.to_csv(filename,index=False)
        pass
    
    def get_facts(self,filename):
        return pd.read_csv(filename)
    
    def get_chunk_similarity_matrix(self):
        pass
    
    def get_linear_sum_assigned_indices(self, threshold):
        pass