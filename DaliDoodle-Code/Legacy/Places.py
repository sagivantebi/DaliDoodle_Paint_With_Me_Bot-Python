import random

from Story_Object import Story_Object


class Places(Story_Object):
    def __init__(self, nlp_engine, answers, place):
        super().__init__(answers, nlp_engine)
        self.place = place
        self.questions = ["Where our story take place?\n"]
        self.city_questions = ["Does the painting depict a major city?"]
        self.gen_place_questions = ["Does the painting depict a general location"]

    def ask_about_object(self):
        sentence = self.nlp_engine.speak_input(random.choice(self.city_questions))
        if sentence == "yes":
            answer = self.nlp_engine.speak_input("great, can you tell me about the city?")
            self.add_answers(answer)

        ners = self.nlp_engine.extract_ners(sentence, "LOC")  ## ners with adjectives
        nouns = self.nlp_engine.extract_nouns(sentence)  ## nouns with adjectives

        place = ":::::"
        self.place = place
        super.set_answers(self.place)
