class Story_Object:
    def __init__(self, nlp_engine, answers):
        self.nlp_engine = nlp_engine
        self.answers = []

    def ask_about_object(self):
        pass

    def get_questions(self):
        return self.questions

    def set_answers(self, answers):
        self.answers = answers

    def add_answers(self, answers):
        self.answers.apeend(answers)

    def remove_answers(self, answers):
        self.answers.remove(answers)
