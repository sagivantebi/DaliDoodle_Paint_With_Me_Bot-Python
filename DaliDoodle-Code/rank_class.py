import json

import pymongo
from bson.json_util import dumps
from pymongo import MongoClient


class RankStat:
    def __init__(self):
        """
         A class to store and manage feedback and statistics related to a ranking or evaluation process.

         Attributes:
             number_of_objs (int): The total number of objects being ranked or evaluated.
             gender (list): List of genders associated with the ranking.
             age (list): List of ages associated with the ranking.
             ease_of_use (list): List of ease of use ratings for the objects.
             recommendation (list): List of recommendation ratings for the objects.
             satisfaction (list): List of satisfaction ratings for the objects.
             overall_experience (list): List of overall experience ratings for the objects.
             issues (list): List of reported issues related to the objects.
             improvements (list): List of suggested improvements for the objects.
             question_feedback (list): List of feedback related to specific questions.
             next_time_question (list): List of responses to the question about next time usage.
             prompt (list): List of prompts used during the evaluation process.
             first_prompt (list): List of first prompts given to participants.
             user_enhance_prompt (list): List of prompts to enhance user input.
             gpt_prompt (list): List of prompts used for GPT-3.5 text generation.
             url (list): List of URLs associated with the objects or evaluation process.
             first_time (list): List of boolean values indicating if it's the user's first time.
             all_prompts (list): List of all prompts used throughout the process.
         """
        self.number_of_objs = 0
        self.gender = []
        self.age = []
        self.ease_of_use = []
        self.recommendation = []
        self.satisfaction = []
        self.overall_experience = []
        self.issues = []
        self.improvements = []
        self.question_feedback = []
        self.next_time_question = []
        self.prompt = []
        self.first_prompt = []
        self.user_enhance_prompt = []
        self.gpt_prompt = []
        self.url = []
        self.first_time = []
        self.all_prompts = []

    def set_stats(self):
        """
           Retrieves and sets statistics and feedback from a MongoDB collection for ranking or evaluation data.

           This method connects to a MongoDB collection, retrieves the ranking data, and populates the attributes
           of the RankStat instance with the collected information.

           Note:
               This method assumes that the MongoDB collection structure matches the attributes of the RankStat class.

           Returns:
               None
           """
        connection_string = "mongodb+srv://omribh:313255242@dalidoodle.lyjrxxl.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(connection_string)
        db = client["ranking_table"]
        collections = db["rank"]

        # Get all objects
        objects = collections.find()

        # # Serialize objects to JSON
        json_objects = dumps(objects)
        json_list = json.loads(json_objects)
        self.number_of_objs = len(json_list)
        for obj in json_list:
            self.gender.append(obj['gender'])
            self.age.append(obj['age'])
            self.ease_of_use.append(obj['ease_of_use'])
            self.recommendation.append(obj['recommendation'])
            self.satisfaction.append(obj['satisfaction'])
            self.overall_experience.append(obj['overall_experience'])
            self.issues.append(obj['issues'])
            self.improvements.append(obj['improvements'])
            self.question_feedback.append(obj['question_feedback'])
            self.next_time_question.append(obj['next_time_question'])
            try:
                self.prompt.append(obj['prompt'])
                self.prompt.append(obj['first_prompt'])
                self.prompt.append(obj['user_enhance_prompt'])
                self.prompt.append(obj['gpt_prompt'])
            except Exception:
                print("there is no prompt to this user")
            try:
                self.prompt.append(obj['first_time'])
                self.url.append(obj['url'])
            except Exception:
                print("there is no url to this user")

    def get_stats_avg(self):
        """
        Calculates the average statistics from the stored feedback and ranking data.

        This method computes the average values for attributes like ease of use, recommendation, satisfaction,
        and overall experience based on the stored feedback data.

        Returns:
            dict: A dictionary containing the calculated average values for different attributes.
        """
        dict_avg = {}

        # avg ease_of_use
        avg_eou = 0
        for eou in self.ease_of_use:
            avg_eou += int(eou)
        dict_avg['ease_of_use'] = avg_eou / self.number_of_objs

        # avg recommendation
        avg_rec = 0
        for rec in self.recommendation:
            avg_rec += int(rec)
        dict_avg['recommendation'] = avg_rec / self.number_of_objs

        # avg satisfaction
        avg_sat = 0
        for sat in self.satisfaction:
            avg_sat += int(sat)
        dict_avg['satisfaction'] = avg_sat / self.number_of_objs

        # avg satisfaction
        avg_exp = 0
        for exp in self.overall_experience:
            avg_exp += int(exp)
        dict_avg['overall_experience'] = avg_exp / self.number_of_objs

        return dict_avg

class Rank:
    """
    A class to store and manage ranking or evaluation data for a single participant's feedback.

    Attributes:
        gender (str): The gender associated with the participant's feedback.
        age (int): The age associated with the participant's feedback.
        ease_of_use (int): The ease of use rating provided by the participant.
        recommendation (int): The recommendation rating provided by the participant.
        satisfaction (int): The satisfaction rating provided by the participant.
        overall_experience (int): The overall experience rating provided by the participant.
        issues (str): Reported issues or problems mentioned by the participant.
        improvements (str): Suggestions for improvements provided by the participant.
        question_feedback (str): Feedback related to specific questions in the evaluation.
        next_time_question (str): Response to the question about using the service next time.
        prompt (str): Prompt used during the evaluation process.
        url (str): URL associated with the participant's feedback.
        first_prompt (str): First prompt given to the participant.
        user_enhance_prompt (str): Prompt to enhance or modify user input.
        gpt_prompt (str): Prompt used for GPT-3.5 text generation.
        first_time (bool): Boolean indicating if it's the participant's first time.
    """
    def __init__(self, request, prompt, url, first_prompt, user_enhance_prompt, gpt_prompt,first_time):
        """
        Initializes a new instance of the Rank class with data provided through a request and prompts.

        Args:
            request (object): The request object containing participant feedback data.
            prompt (str): The main prompt used during the evaluation process.
            url (str): URL associated with the participant's feedback.
            first_prompt (str): First prompt given to the participant.
            user_enhance_prompt (str): Prompt to enhance or modify user input.
            gpt_prompt (str): Prompt used for GPT-3.5 text generation.
            first_time (bool): Boolean indicating if it's the participant's first time.
        """
        self.gender = request.form['gender']
        self.age = request.form['age']
        self.ease_of_use = request.form['ease-of-use']
        self.recommendation = request.form['recommendation']
        self.satisfaction = request.form['satisfaction']
        self.overall_experience = request.form['overall-experience']
        self.issues = request.form['issues']
        self.improvements = request.form['improvements']
        self.question_feedback = request.form['question-feedback']
        self.next_time_question = request.form['next-time-question']
        self.prompt = prompt
        self.url = url
        self.first_prompt = first_prompt
        self.user_enhance_prompt = user_enhance_prompt
        self.gpt_prompt = gpt_prompt
        self.first_time = first_time


    def print_rank(self):
        """
        Prints the participant's feedback and ranking data to the console.

        This method prints the various attributes of the Rank instance, providing an overview
        of the participant's feedback and ranking data.

        Returns:
            None
        """
        print(self.gender)
        print(self.age)
        print(self.ease_of_use)
        print(self.recommendation)
        print(self.satisfaction)
        print(self.overall_experience)
        print(self.issues)
        print(self.improvements)
        print(self.question_feedback)
        print(self.next_time_question)
        print(self.prompt)
        print(self.url)
        print(self.first_prompt)
        print(self.user_enhance_prompt)
        print(self.gpt_prompt)
        print(self.first_time)

    def add_new_rank(self):
        """
         Adds the participant's feedback and ranking data to a MongoDB collection.

         This method connects to a MongoDB collection, inserts the participant's feedback data
         as a new document, and stores it in the database for analysis.

         Returns:
             None
         """
        connection_string = "mongodb+srv://omribh:313255242@dalidoodle.lyjrxxl.mongodb.net/?retryWrites=true&w=majority"
        client = pymongo.MongoClient(connection_string)
        db = client["ranking_table"]
        collections = db["rank"]
        new_rank = {
            'gender': self.gender,
            'age': int(self.age),
            'ease_of_use': int(self.ease_of_use),
            'recommendation': int(self.recommendation),
            'satisfaction': int(self.satisfaction),
            'overall_experience': int(self.overall_experience),
            'issues': self.issues,
            'improvements': self.improvements,
            'question_feedback': self.question_feedback,
            'next_time_question': self.next_time_question,
            'prompt': self.prompt,
            'first_prompt': self.first_prompt,
            'user_enhance_prompt': self.user_enhance_prompt,
            'gpt_prompt': self.gpt_prompt,
            'first_time': self.first_time
        }
        collections.insert_one(new_rank)

