import os
import random
import shutil

import requests

from Hard_coded import GPE_CONVO, PER_CONVO, DATE_CONVO, NOUN_CONVO
from Story_Teller import Story_Teller

OIL_STYLE = 1
PORTRAIT_STYLE = 2
ABSTRACT_STYLE = 3
SURREAL_STYLE = 4
WATERCOLOR_STYLE = 5

CASE_INIT_PROMPT = 1
CASE_PAINTING_TYPE_PROMPT = 2
CASE_INIT_USER_PROMPT = 3
CASE_CONVERSATION_LOOP_PROMPT = 4
CASE_ADD_TO_PROMPT = 5
CASE_FINAL_PROMPT = 6
CASE_PRINT_PROMPT = 7
CASE_PASS_EDIT = "SUCCESS"
CASE_ADD_TO_PROMPT_LOOP = 12
CASE_SHOULD_KEEP_WORKING_ON_DESC = 8
CASE_ADD_OR_REPLACE = 9
CASE_WHAT_TO_REPLACE = 10
CASE_REPLACE_WITH = 11
INIT_CONVERSATION = 0

SUCCESS = "SUCCESS"

#documentation for this code:

class Web_Free_Story(Story_Teller):
    """
    This is the class that is used to generate the story for the user. It is a subclass of Story_Teller.
    """
    def __init__(self, nlp_engine):
        super().__init__(nlp_engine)

    def __add_type_img(self, prompt):
        """
            get the image type the user wants to paint
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
        get the output sentence to the user using the nlp_engine object
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


    def gpt_or_mask_extend_sentence(self, running_sentence, extend_type="MASK"):
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
    """"
    initial prompt for the user
    """
    def case_init_prompt(self, user_answer=None):
        return 'Let\'s start working on our Painting together\n What would you like to draw?\n'


    def case_painting_type_prompt(self, user_answer):
        """
           get the painting type from the user
        """
        if user_answer == 'n':
            self.answer_counter = CASE_INIT_PROMPT
            return 'Sorry, but you cannot skip this step.\n Please tell me what you would like to draw ?\n'

        processed_answer = self.nlp_engine.check_input(user_answer)
        # removed it because of common problems
        #
        # processed_answer = self.nlp_engine.QA_extract_user_descriptions(processed_answer)
        self.set_prompt(processed_answer)
        self.set_first_prompt(processed_answer)
        # return "What visual style should our painting be? \n1. Oil \n2. Portrait \n3. Abstract \n4. Surrealism \n5. Watercolour\n"
        # sagiv : I added this because I cant see this page anymore :)3
        if user_answer == "":
            self.answer_counter += 1
        self.answer_counter += 1
        self.initial_ners_and_nouns()
        self.init_when_to_stop()

        # initialize the counter for how many times we will ask the user to add to the prompt
        if self.check_len_ners_nouns() and self.when_to_stop:
            # get a random named entity or noun
            ner_or_noun = self.get_random_ner_or_noun()
            self.minus_when_to_stop()
            self.minus_when_to_stop()
            # get a question for the NER/noun and ask the user
            ner_or_noun_user_question = self.make_descriptions(ner_or_noun)
            return ner_or_noun_user_question
        elif not self.check_len_ners_nouns() and self.when_to_stop:
            self.answer_counter -= 2
            return "It appears your description lacks content, please try again"

    def case_init_user_prompt(self, user_answer):
        """
        Generates a user prompt based on the provided user_answer, which indicates the selected painting style.

        Args:
            user_answer (str): The user's response indicating the selected painting style.

        Returns:
            str: A user prompt or instruction for the selected painting style.

        Note:
            The function handles input validation and generates prompts based on different painting styles.

        Example:
            generator = PaintingPromptGenerator()
            prompt = generator.case_init_user_prompt("1")
            print(prompt)  # Output: "Oil painting of"
        """
        try:
            type_img = int(user_answer)  # Convert user input to integer
        except Exception:
            self.answer_counter -= 1
            return "Please enter a number between 1-5\n"  # Input validation message

        # Dictionary mapping painting style constants to descriptive strings
        switcher = {
            OIL_STYLE: "Oil painting of",
            PORTRAIT_STYLE: "Portrait painting of",
            ABSTRACT_STYLE: "Abstract painting of",
            SURREAL_STYLE: "Surrealism painting of",
            WATERCOLOR_STYLE: "Watercolour painting of",
        }

        # Set the painting style description based on the user's choice or use a default description
        self.set_type(switcher.get(type_img, "Realistic painting of"))

        # Initialize named entities and nouns for the prompt
        self.initial_ners_and_nouns()

        # Initialize the counter for how many times prompts will be generated
        if self.check_len_ners_nouns() and self.when_to_stop:
            ner_or_noun = self.get_random_ner_or_noun()  # Get a random named entity or noun
            self.minus_when_to_stop()  # Decrement the prompt generation threshold
            ner_or_noun_user_question = self.make_descriptions(ner_or_noun)  # Generate a question for the NER/noun
            return ner_or_noun_user_question  # Return the generated prompt
        elif not self.check_len_ners_nouns() and self.when_to_stop:
            self.answer_counter -= 1
            return "It appears your description lacks content, please try again"  # Incomplete prompt message

    def case_conversation_loop_prompt(self, user_answer):
        """
        Loops x amount of times to get more descriptions from the user

        Args:
            user_answer (str): The user's response indicating whether they want to add content to the prompt.

        Returns:
            str: A prompt or instruction for the user to engage in a conversation to enhance the existing prompt.

        Example:
            generator = ConversationPromptGenerator()
            prompt = generator.case_conversation_loop_prompt("y")
            print(prompt)  # Output: "Abstract painting of a serene landscape with vibrant colors..."

        Note:
            The function checks user preferences for prompt enhancement, performs correction on the prompt,
            facilitates a conversation loop for adding content, and guides users through the process.
        """
        # Check if the user wants to continue adding to the prompt
        if user_answer != "":
            check_handle = self.conversation_loop_prompt_helper(user_answer)
            corrected_answer = self.nlp_engine.check_input(self.get_prompt())
            self.set_prompt(corrected_answer)
        else:
            check_handle = self.conversation_loop_prompt_helper(user_answer)

        # Generate prompts by asking about objects or named entities and adding to the prompt
        if self.check_len_ners_nouns() and self.when_to_stop:
            self.answer_counter -= 1
            ner_or_noun = self.get_random_ner_or_noun()
            self.minus_when_to_stop()
            noun_description = self.make_descriptions(ner_or_noun)
            return noun_description
        else:
            self.set_user_enhance_prompt(self.get_prompt())
            return f'Alright, so we have the prompt: \n"{self.get_prompt()}",' \
                   f'\n Would you like me to try ENHANCING it a little? (y/n)\n'

    def gpt_or_mask_extend_sentence_new(self, running_sentence, mask_or_gpt):
        """
        Extends the sentence with either MASK or GPT
        :param running_sentence: the current sentence to enhance
        :param mask_or_gpt: use mask bert or gpt
        :return: extended sentence after enhancement
        """
        extend_sentence = ""
        if mask_or_gpt == "MASK":
            extend_sentence = (self.nlp_engine.extend_sentence_with_MASK(running_sentence))
        elif mask_or_gpt == "GPT":
            extend_sentence = (self.nlp_engine.extend_sentence(running_sentence))
        return extend_sentence

    def case_add_to_prompt(self, user_answer):
        """
        Adds to the prompt with sentence enhancement using the nlp engine
        :param user_answer:
        :return:
        """
        if user_answer.lower() == "y":
            running_sentence = self.nlp_engine.enhance_sentence(self.get_prompt())
            self.set_gpt_prompt(running_sentence)
            self.set_model_extend_prompt(running_sentence)
            return f'Okay, so I think this could be a good fit for the prompt:\n"{running_sentence}"\n Do you like this version of the prompt?\n'
        elif user_answer.lower() == "n":
            # pass case six
            self.skip_case()
            return "Alright! \n" + self.get_prompt() + "\nWould you like me to PAINT it for you? (y/n)\n"
        else:
            if(user_answer!=""):
                self.answer_counter -= 1
            return "Please enter \"y\" or \"n\"."

    def case_final_prompt(self, user_answer):
        """
        Final prompt for the user - display the final version of the prompt
        and decide whether to paint or not
        :param user_answer:
        :return:
        """
        # next case
        self.answer_counter = CASE_ADD_TO_PROMPT_LOOP - 1
        # if the user likes the prompt set the prompt to the extended version
        if user_answer.lower() == "great":
            self.answer_counter = CASE_PRINT_PROMPT
            self.set_prompt(self.get_model_extend_prompt())
            return self.case_print_prompt("y")
        # if the user wants to add more to the prompt get more and add it
        elif user_answer.lower() == "more":
            self.set_prompt(self.get_model_extend_prompt())
            running_sentence = self.nlp_engine.enhance_sentence(self.get_prompt())
            self.set_gpt_prompt(running_sentence)
            self.set_model_extend_prompt(running_sentence)
            return f'Okay, so I think this could be a good fit for the prompt:\n"{running_sentence}"\n Do you like this version of the prompt?\n'
        # if the user wants to try again get a new prompt and ask if they like it
        elif user_answer.lower() == "again":
            running_sentence = self.nlp_engine.enhance_sentence(self.get_prompt())
            self.set_gpt_prompt(running_sentence)
            self.set_model_extend_prompt(running_sentence)
            return f'Okay, so I think this could be a good fit for the prompt:\n"{running_sentence}"\n Do you like this version of the prompt?\n'
        # if the user wants to skip the enhancement process
        elif user_answer.lower() == "original":
            self.answer_counter = CASE_PRINT_PROMPT
            return self.case_print_prompt("y")



    def case_print_prompt(self, user_answer):
        """
        Print the final version of the prompt and ask if the user wants to save it
        :param user_answer:
        :return:
        """
        if user_answer.lower() == "y":
            url = self.nlp_engine.text_to_img_for_web_dale2(self.get_type() + " " + self.get_prompt())
            self.set_url(url[0])
            print(url)
            return_val = self.save_img(url[0], self.get_prompt())
            if return_val == True:
                print("saved")
            else:
                print("not saved")
            if not url:
                return "Sorry, our server busy right now, please try again later."
            return {"url": url, "question": "Would you like to draw another painting together? (y/n)\n", \
                    "description": self.get_prompt()}
        elif user_answer.lower() == "n":
            # return "Do you want to keep working on the description of the picture? (y/n)\n"
            return "Would you like to draw another painting together? (y/n)\n"
        else:
            if(user_answer!=""):
                self.answer_counter -= 1
            return "Please enter \"y\" or \"n\"."

    def case_should_keep_working_on_desc(self, user_answer):
        """
        Ask the user if they want to keep working on the description
        :param user_answer:
        :return:
        """
        if user_answer.lower() == "n":
            # return this string to the main and in the main there is a case that will redirect to the ranking page
            # Exit from the conversation and moving to the ranking page
            return "redirect to ranking"
        elif user_answer.lower() == "y":
            self.reset_prompts()
            self.answer_counter = CASE_INIT_PROMPT
            return "Let\'s start working on our Painting together\n What would you like to draw?\n"
            # return "Would you like to add or replace something in the description? (add/replace)\n"
        else:
            if(user_answer!=""):
                self.answer_counter -= 1
            return "Please enter \"y\" or \"n\"."

    def case_add_or_replace(self, user_answer):
        """
        Ask the user if they want to add or replace something in the description
        :param user_answer:
        :return:
        """
        if user_answer.lower() == "a" or user_answer.lower() == "add":
            # Bring the user back to case 4 and the loop to run one more time
            self.answer_counter = 2
            self.initial_ners_and_nouns()
            self.init_when_to_stop()
            # initialize the counter for how many times we will ask the user to add to the prompt
            if self.check_len_ners_nouns() and self.when_to_stop:
                # get a random named entity or noun
                ner_or_noun = self.get_random_ner_or_noun()
                self.minus_when_to_stop()
                # get a question for the NER/noun and ask the user
                ner_or_noun_user_question = self.make_descriptions(ner_or_noun)
                return ner_or_noun_user_question
            elif not self.check_len_ners_nouns() and self.when_to_stop:
                self.answer_counter -= 2
                return "It appears your description lacks content, please try again"
        elif user_answer.lower() == "r" or user_answer.lower() == "replace":
            return "What would you like to replace in the sentence?\n" + self.get_prompt() + "\n"
        else:
            if(user_answer!=""):
                self.answer_counter -= 1
            return "Please enter \"add\" or \"remove\"."

    def case_what_to_replace(self, user_answer):
        """
        Ask the user what he wants to replace in the description
        :param user_answer:
        :return:
        """
        # use the qa engine to extract the replacement word from what the user asked to replace
        what_to_replace = self.nlp_engine.qa_engine.QA_extract("what should be replaced?", user_answer)
        if what_to_replace not in self.get_prompt():
            nouns = self.nlp_engine.extract_nouns(self.get_prompt())
            noun = [n for n in nouns if n in what_to_replace]
            if len(noun) == 1:
                noun = noun[0]
                if noun in self.get_prompt():
                    noun_tree = self.nlp_engine.get_target_noun_tree(noun, self.get_prompt())
                    print(noun_tree)
                    self.set_what_to_replace(noun_tree)
                    return f'Alright,with what would you like to replace "{noun_tree}"?\n'
            self.answer_counter = 9
            return f'"{what_to_replace}" is not in the sentence , please try again\n What would you like to replace in the sentence?\n '
        self.set_what_to_replace(what_to_replace)
        return f'Alright,with what would you like to replace "{what_to_replace}"?\n'


    def case_add_to_sentence_loop(self, user_answer):
        """
        ask if the user wants to enhcance the sentence - if yes, enhance the sentence and ask again.
        if the user doesn't want to enhance the sentence, print the sentence and ask if the user wants to keep working on the description
        if the user likes the sentence, print the sentence and ask if the user wants to add more to the sentence

        :param user_answer:
        :return:
        """
        # the user wants to add more to the sentence
        if user_answer.lower() == "more":
            self.set_prompt(self.get_model_extend_prompt())
            enhanced_sentence = self.nlp_engine.enhance_sentence(self.get_prompt())
            self.set_model_extend_prompt(enhanced_sentence)
            self.answer_counter = CASE_FINAL_PROMPT - 1
            return f'Alright, so I think this could be a great addition to the prompt:\n"{enhanced_sentence}"\n Do you like this new addition to the prompt?\n'
        # the user doesn't want to add more to the sentence
        elif user_answer.lower() == "great":
            self.set_prompt(self.get_model_extend_prompt())
            self.answer_counter = CASE_PRINT_PROMPT
            return self.case_print_prompt("y")
        elif user_answer.lower() == "original":
            self.answer_counter = CASE_PRINT_PROMPT
            return self.case_print_prompt("y")
        # the user wants the program to generate a new addition to the sentence
        elif user_answer.lower() == "again":
            running_sentence = self.nlp_engine.enhance_sentence(self.get_prompt())
            self.set_model_extend_prompt(running_sentence)
            self.answer_counter = CASE_FINAL_PROMPT - 1
            return f'Then let\'s try again, I think this could be a good fit for the prompt:\n" {running_sentence}"\n Do you like this version of the prompt?\n'
        # the user wants to skip the addition to the sentence
        elif user_answer.lower() == "skip" or user_answer.lower() == "no":
            self.answer_counter = CASE_PRINT_PROMPT - 1
            return "Alright! we finished editing the prompt, and the final result is: \n" + self.get_prompt() + "\nWould you like me to PAINT it for you? (y/n)\n"
        else:
            self.answer_counter = CASE_FINAL_PROMPT - 1
            return "Please enter Skip or Again\n"

    def conversation_loop_prompt_helper(self, user_answer):
        """
        This function is a helper function for the conversation loop.
        It handles the cases of the conversation loop that are related to the prompt.

        :param user_answer:
        :return:
        """
        running_sentence = self.get_prompt()
        if user_answer == "":
            return False
        elif user_answer.lower() == "n" or user_answer.lower() == "no":
            return SUCCESS
        elif self.get_last_ner_or_noun_talked_about()[0] == "ner":
            replacement = self.nlp_engine.add_descriptions_QA(user_answer, self.get_last_ner_or_noun_talked_about()[1])
            running_sentence = running_sentence.replace(self.get_last_ner_or_noun_talked_about()[1], replacement)
        elif self.get_last_ner_or_noun_talked_about()[0] == "noun":
            replacement = self.nlp_engine.add_descriptions_QA(user_answer, self.get_last_ner_or_noun_talked_about()[1])
            running_sentence = running_sentence.replace(self.get_last_ner_or_noun_talked_about()[1], replacement)
        self.set_prompt(running_sentence)
        return SUCCESS

    def get_ner_description_new(self, ner):
        """
        This function returns a random prompt that is contains the ner that was given as a parameter.

        :param ner:
        :return:
        """
        ners = self.nlp_engine.extract_ners_tokens(self.get_prompt())
        spacy_ner = [n for n in ners if n.text == ner][0]
        entity_label = spacy_ner.label_
        if entity_label == "GPE":
            rand_prompt = random.choice(GPE_CONVO).replace("<ner>", spacy_ner.text)
        elif entity_label == "PERSON":
            rand_prompt = random.choice(PER_CONVO).replace("<ner>", spacy_ner.text)
        elif entity_label == "DATE":
            rand_prompt = random.choice(DATE_CONVO).replace("<ner>", spacy_ner.text)
        else:
            rand_prompt = f"I don't know anything about {spacy_ner.text}, can you add more details?"
        return rand_prompt

    def make_descriptions(self, ner_or_noun):
        """
        This function returns a random prompt that is contains the ner or noun that was given as a parameter.
        :param ner_or_noun:
        :return:
        """
        if ner_or_noun[0] == "ner":
            return self.get_ner_description_new(ner_or_noun[1])
        elif ner_or_noun[0] == "noun":
            random_output_prompt = random.choice(NOUN_CONVO)
            random_output_prompt = random_output_prompt.replace("<noun>", ner_or_noun[1])
            return random_output_prompt
        else:
            return None

    def initial_ners_and_nouns(self):
        """
        This function initializes the ners and nouns of the prompt.
        :return:
        """
        self.set_ners([ner for ner in self.nlp_engine.extract_ners(self.get_prompt())])
        # self.set_nouns([noun for noun in self.nlp_engine.extract_noun_chunks(self.get_prompt())])
        self.set_nouns([noun for noun in self.nlp_engine.extract_nouns(self.get_prompt())])

    def check_len_ners_nouns(self):
        """
        This function checks if there are any ners or nouns in the prompt.
        :return:
        """
        if len(self.get_ners()) == 0 and len(self.get_nouns()) == 0:
            return False
        return True

    def skip_case(self):
        """"
        This function is used to skip a case in the conversation loop."""
        self.answer_counter += 1

    def create_story(self, user_answer):
        """
        This function is used to create a story from the prompt.
        :param user_answer:
        :return:
        """
        self.stack_user_inputs.append(user_answer)
        # TODO: handle refresh/empty user_answer
        if user_answer != "":
            self.skip_case()
        if self.answer_counter == CASE_INIT_USER_PROMPT:
            self.skip_case()
        switcher = {
            CASE_INIT_PROMPT: self.case_init_prompt,
            CASE_PAINTING_TYPE_PROMPT: self.case_painting_type_prompt,
            CASE_INIT_USER_PROMPT: self.case_init_user_prompt,
            CASE_CONVERSATION_LOOP_PROMPT: self.case_conversation_loop_prompt,
            CASE_ADD_TO_PROMPT: self.case_add_to_prompt,
            CASE_FINAL_PROMPT: self.case_final_prompt,
            CASE_PRINT_PROMPT: self.case_print_prompt,
            CASE_SHOULD_KEEP_WORKING_ON_DESC: self.case_should_keep_working_on_desc,
            CASE_ADD_OR_REPLACE: self.case_add_or_replace,
            CASE_WHAT_TO_REPLACE: self.case_what_to_replace,
            CASE_REPLACE_WITH: self.case_replace_with,
            CASE_ADD_TO_PROMPT_LOOP: self.case_add_to_sentence_loop
        }
        func = switcher.get(self.answer_counter, lambda: None)
        need_to_return = func(user_answer)
        return need_to_return

    def decrease_answer_counter(self):
        """
        This function decreases the answer counter by 2.
        :return:
        """
        if self.answer_counter <= 0:
            return
        self.answer_counter -= 2

    def last_answer(self):
        """
        This function returns the last answer that was given by the user.
        :return:
        """
        if len(self.stack_user_inputs) != 0:
            return self.stack_user_inputs.pop()
        return None

    def clear_all_descriptions(self):
        self.stack_user_inputs = self.stack_user_inputs[:2]

    def save_img(self, url, prompt):
        """
        This function saves an image from a url.
        :param url:
        :param prompt:
        :return:
        """
        #clean prompt
        prompt = prompt.split(" ")
        prompt = "_".join(prompt)
        prompt = prompt.split(".")
        prompt = "".join(prompt)
        rand_num = str(random.randint(0, 10000))
        current_dir = os.getcwd()
        folder_path = os.path.join(current_dir, "static")
        folder_path = os.path.join(folder_path, "Imgs")
        file_name = os.path.join(folder_path, prompt + rand_num + ".jpeg")
        try:
            res = requests.get(url, stream=True)
            if res.status_code == 200:
                with open(file_name, 'wb') as f:
                    shutil.copyfileobj(res.raw, f)
                self.nlp_engine.speak_print('Image successfully Downloaded')
                return True
            else:
                self.nlp_engine.speak_print('Image Could not be retrieved')
                return False
        except Exception:
            print("There was a problem saving")
            return False
    def reset_prompts(self):
        self.set_prompt("")
        self.set_first_prompt("")
        self.set_user_enhance_prompt("")
        self.set_gpt_prompt("")