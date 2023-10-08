import random

from Hard_coded import GPE_CONVO, PER_CONVO, DATE_CONVO, NOUN_CONVO
from Story_Teller import Story_Teller


ALRIGHT_FINISH_PROMPT = "Alright! we finished editing the prompt, and the final result is: \n"
class Free_Story(Story_Teller):
    """
Class `Free_Story`:
-------------------
This class represents a story generation system that allows the user to create a painting-related story.
It extends the `Story_Teller` base class.

Attributes:
-----------
- `ALRIGHT_FINISH_PROMPT`: A constant string indicating the completion of the prompt editing process.

Methods:
--------
1. `__init__(self, nlp_engine)`
    Constructor method for the `Free_Story` class.

    Parameters:
    - `nlp_engine`: An instance of a natural language processing engine.

2. `__add_type_img(self, prompt) -> str`
    Private method to add the type of painting to the given prompt.

    Parameters:
    - `prompt`: The initial prompt text.

    Returns:
    - A modified prompt including the type of painting.

3. `get_ner_description(self, ners) -> str`
    Generates a description for a randomly selected named entity recognition (NER) entity.

    Parameters:
    - `ners`: A list of Named Entity Recognition (NER) entities.

    Returns:
    - A description string for the selected NER entity.

4. `gpt_or_mask_extend_sentence(self, running_sentence, extend_type="MASK") -> str`
    Extends a sentence using either MASK-based or GPT-based sentence extension.

    Parameters:
    - `running_sentence`: The sentence to be extended.
    - `extend_type`: The type of extension to perform ("MASK" or "GPT").

    Returns:
    - The extended sentence.

5. `enhance_text_new(self)`
    Enhances the text by adding descriptions to nouns and named entities in the sentence.

6. `create_story(self)`
    Creates a story by interacting with the user to gather prompts and enhancing the text.
"""
    def __init__(self, nlp_engine):
        super().__init__(nlp_engine)

    def __add_type_img(self, prompt):
        """
        Add Type of Painting to the Prompt

        Private method that prompts the user to select a type of painting and adds it to the provided prompt.

        Parameters:
            prompt (str): The initial prompt text.

        Returns:
            str: The modified prompt including the selected type of painting.
        """
        type_img = self.nlp_engine.speak_input("What Type of painting do you want our Paint to be:",
                                               "\n1. Oil \n2. Portrait \n3. Abstract \n4. Surrealism \n5. Watercolour\n")
        try:
            type_img = int(type_img)
        except Exception:
            return
        if type_img == 1:
            return "Oil painting of " + prompt
        elif type_img == 2:
            return "Portait painting of " + prompt
        elif type_img == 3:
            return "Abstract painting of " + prompt
        elif type_img == 4:
            return "Surrealism painting of " + prompt
        elif type_img == 5:
            return "Watercolour painting of " + prompt
        else:
            return "Realistic painting of " + prompt

    def get_ner_description(self, ners):
        """
    Generate Description for a Named Entity

    Generates a description for a randomly selected Named Entity Recognition (NER) entity.

    :param: ners (list): A list of Named Entity Recognition (NER) entities.

    Returns:
        str: A description for the selected NER entity.
    """
        random_entity = random.choice(ners)
        entity_label = random_entity.label_
        if entity_label == "GPE":
            rand_prompt = random.choice(GPE_CONVO).replace("<ner>", random_entity.text)
        elif entity_label == "PERSON":
            rand_prompt = random.choice(PER_CONVO).replace("<ner>", random_entity.text)
        elif entity_label == "DATE":
            rand_prompt = random.choice(DATE_CONVO).replace("<ner>", random_entity.text)
        else:
            rand_prompt = f"I don't quite know what {random_entity.text} is, can you add details?"
        running_input = self.nlp_engine.speak_input(rand_prompt)

        if running_input == "" or running_input.lower() == "no":
            return None
        replacement = running_input + " " + random_entity.root.text
        return replacement

    def gpt_or_mask_extend_sentence(self, running_sentence, extend_type= "MASK"):
        """
       Extend Sentence using MASK or GPT

       Extends the given sentence using either MASK-based or GPT-based sentence extension.

       Parameters:
           running_sentence (str): The sentence to be extended.
           extend_type (str): The type of extension to perform ("MASK" or "GPT").

       Returns:
           str: The extended sentence.
    """
        extend_sentence = ""
        self.nlp_engine.speak_print("I feel like this could be a good fit to the sentence:\n")
        if extend_type == "MASK":
            extend_sentence = (self.nlp_engine.extend_sentence_with_MASK(running_sentence))
        elif extend_type == "GPT":
            extend_sentence = (self.nlp_engine.extend_sentence(running_sentence))
        self.nlp_engine.speak_print(extend_sentence)
        want_the_new = self.nlp_engine.speak_input("Do you like it? ", "(y/n)\n")
        if want_the_new.lower() == "y":
            return extend_sentence
        return running_sentence

    def enhance_text_new(self):
        """
    Enhance Text by Adding Descriptions

    Enhances the text by adding descriptions to nouns and named entities in the sentence.
    """
        # this section is just for splitting the type out of the enhance process
        nlp = self.nlp_engine
        running_sentence = self.get_prompt()
        nouns_ners_list = [("ners", ners) for ners in nlp.extract_ners(running_sentence)]
        nouns_ners_list.extend([("nouns", nouns) for nouns in  nlp.extract_nouns(running_sentence)])
        till_stop = min(len(nouns_ners_list), 3)
        while till_stop > 0:
            random_nouns_ners = random.choice(nouns_ners_list)
            nouns_ners_list.remove(random_nouns_ners)
            # ask from the user to add description to the NERS
            if random_nouns_ners[0] == "ners":
                rand_prompt = self.get_ner_description(random_nouns_ners[1])
                if rand_prompt is not None:
                    running_sentence = running_sentence.replace(random_nouns_ners[1], rand_prompt)
            # ask from the user to add description to the NOUNS
            elif random_nouns_ners[0] == "nouns":
                random_output_prompt = random.choice(NOUN_CONVO)
                random_output_prompt = random_output_prompt.replace("<noun>", random_nouns_ners[1])
                running_input = self.nlp_engine.speak_input(random_output_prompt)
                if running_input == "" or running_input.lower() == "no":
                    continue
                replacement = self.nlp_engine.add_descriptions_QA(running_input, random_nouns_ners[1])
                running_sentence = running_sentence.replace(random_nouns_ners[1], replacement)
            till_stop -= 1
            if till_stop == 1:
                choice = self.nlp_engine.speak_input(
                    f'Alright, so we have the prompt: \n{running_sentence},\n would you like to keep adding to it?', '(y/n)\n')
                if choice.lower() == "y":
                    running_sentence = self.gpt_or_mask_extend_sentence(running_sentence, "MASK")
                elif choice.lower() == "n":
                    till_stop = 0
        # if it reached maximum iterations of adding it
        self.set_prompt(running_sentence)



    def create_story(self):
        """
    Create a Story

    Initiates the story creation process by collecting prompts, adding a painting type, and enhancing the text.
    """
        self.set_prompt(self.nlp_engine.speak_input('Let\'s start our Painting together\n What would you like to draw?\n') + ".")
        self.set_type(self.__add_type_img(""))
        self.enhance_text_new()




