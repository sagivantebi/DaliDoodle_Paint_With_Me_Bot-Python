import random

from Story_Object import Story_Object


class People(Story_Object):
    def __init__(self, nlp_engine, answers, people):
        super().__init__(answers, nlp_engine)
        self.people = people
        self.questions = ["Who are the people in the story?\n"]

    def ask_about_object(self):
        sentence = self.nlp_engine.speak_input(random.choice(self.questions))
        # need to take the ner ot the Place using NLP Engine
        people = ":::::"
        self.people = people
        super.set_answers(self.people)
