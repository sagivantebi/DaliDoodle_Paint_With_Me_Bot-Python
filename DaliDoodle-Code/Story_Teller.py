import random
from Hard_coded import CONNECTION_WORDS

INIT_CONVERSATION = 1
INIT_STOP = 3


class Story_Teller:
    """
     A class that facilitates storytelling using a natural language processing engine (NLP).

     Attributes:
         nlp_engine (object): The NLP engine used for text generation and analysis.
         __prompt (str): The current prompt being constructed for interaction.
         __first_prompt (str): The initial prompt used to start a conversation.
         __user_enhance_prompt (str): Prompt to enhance or modify user input.
         __gpt_prompt (str): The prompt used for GPT-3.5 text generation.
         __url (str): URL for referencing external resources like images or documents.
         painting_type (str): Type of painting associated with the story.
         __img_path (str): File path to the image related to the story.
         answer_counter (int): Counter to keep track of the number of interactions.
         ners (list): Named entities recognized from the conversation.
         nouns (list): Nouns extracted from the conversation.
         when_to_stop (int): Counter for deciding when to stop the conversation.
         last_ner_or_noun_talked_about (str): Last named entity or noun discussed.
         model_extend_prompt (str): Additional prompt to extend the model's response.
         what_to_replace (str): Text to be replaced in the user's input.
         with_what_to_replace (str): Text to replace 'what_to_replace' with.
         stack_user_inputs (list): Stack to store previous user inputs.
     """
    def __init__(self, nlp_engine):
        """
               Initializes a new instance of the Story_Teller class.

               Args:
                   nlp_engine (object): The NLP engine to be used for text generation and analysis.
               """
        self.nlp_engine = nlp_engine
        self.__prompt = ""
        self.__first_prompt = ""
        self.__user_enhance_prompt = ""
        self.__gpt_prompt = ""
        self.__url = ""
        self.painting_type = ""
        self.__img_path = ""
        self.answer_counter = INIT_CONVERSATION
        self.ners = []
        self.nouns = []
        self.when_to_stop = INIT_STOP
        self.last_ner_or_noun_talked_about = None
        self.model_extend_prompt = ""
        self.what_to_replace = ""
        self.with_what_to_replace = ""
        self.stack_user_inputs = []

    def init_when_to_stop(self):
        self.when_to_stop = INIT_STOP
    def get_with_what_to_replace(self):
        return self.with_what_to_replace

    def get_nlp_engine(self):
        return self.nlp_engine

    def set_with_what_to_replace(self, with_what_to_replace):
        self.with_what_to_replace = with_what_to_replace
    def get_img_path(self):
        return self.__img_path
    def set_img_path(self, img_path):
        self.__img_path = img_path

    def get_what_to_replace(self):
        return self.what_to_replace
    def set_what_to_replace(self, what_to_replace):
        self.what_to_replace = what_to_replace

    def get_model_extend_prompt(self):
        return self.model_extend_prompt
    def set_model_extend_prompt(self, model_extend_prompt):
        self.model_extend_prompt = model_extend_prompt

    def reset_counter(self):
        self.answer_counter = INIT_CONVERSATION
        self.when_to_stop = INIT_STOP

    def set_user_counter(self, counter):
        self.answer_counter = counter

    def get_answer_counter(self):
        return self.answer_counter

    def minus_when_to_stop(self):
        self.when_to_stop -= 1

    def get_ners(self):
        return self.ners
    def get_nouns(self):
        return self.nouns
    def set_ners(self, ners):
        self.ners = ners
    def set_nouns(self, nouns):
        self.nouns = nouns


    def get_last_ner_or_noun_talked_about(self):
        return self.last_ner_or_noun_talked_about

    def set_last_ner_or_noun_talked_about(self, ner_or_noun):
        self.last_ner_or_noun_talked_about = ner_or_noun
        
    def get_random_ner_or_noun(self):
        """
          Selects a random named entity (NER) or noun from the available NERs and nouns.

          Returns:
              tuple or None: A tuple containing the type of the selected entity ("ner" or "noun")
                             and the chosen entity. Returns None if no entities are available.
          """
        choice = random.choice(["ners", "nouns"])
        ners = self.get_ners()
        nouns = self.get_nouns()
        nouns = [x for x in nouns if x not in ners]
        if choice == "ners":
            if len(ners) != 0:
                ner = random.choice(ners)
                self.set_ners([x for x in ners if x != ner])
                self.set_last_ner_or_noun_talked_about(("ner",ner))
                return ("ner",ner)
            elif len(nouns) != 0:
                noun = random.choice(nouns)
                self.set_nouns([x for x in nouns if x != noun])
                self.set_last_ner_or_noun_talked_about(("noun",noun))
                return ("noun",noun)
            else:
                return None
        else:
            if len(nouns) != 0:
                noun = random.choice(nouns)
                self.set_nouns([x for x in nouns if x != noun])
                self.set_last_ner_or_noun_talked_about(("noun",noun))
                return ("noun",noun)
            elif len(ners) != 0:
                ner = random.choice(ners)
                self.set_ners([x for x in ners if x != ner])
                self.set_last_ner_or_noun_talked_about(("ner",ner))
                return ("ner",ner)
            else:
                return None
    def get_type(self):
        return self.painting_type
    def set_type(self, painting_type):
        self.painting_type = painting_type

    def set_prompt(self, prompt):
        self.__prompt = prompt

    def get_prompt(self):
        return self.__prompt

    def set_first_prompt(self, prompt):
        self.__first_prompt = prompt

    def get_first_prompt(self):
        return self.__first_prompt

    def set_user_enhance_prompt(self, prompt):
        self.__user_enhance_prompt = prompt

    def get_user_enhance_prompt(self):
        return self.__user_enhance_prompt

    def set_gpt_prompt(self, prompt):
        self.__gpt_prompt = prompt

    def get_gpt_prompt(self):
        return self.__gpt_prompt

    def set_url(self, url):
        self.__url = url

    def get_url(self):
        return self.__url

    def set_img_path(self, img_path):
        self.__img_path = img_path

    def get_img_path(self):
        return self.__img_path

    def create_story(self, user_answer):
        pass

    def enhance_text(self):
        pass

    def check_in_list_to_replace(self, list_to_check, prompt_list):
        """
        Checks if words from the provided list are present in the prompt list for replacement.

        Args:
            list_to_check (list): A list of words to check for replacement.
            prompt_list (list): A list of words in the prompt for potential replacement.

        Returns:
            tuple: A tuple containing a boolean value indicating whether a replacement word was found
                   and the word to be replaced. Returns (False, None) if no suitable word is found.
        """
        if len(list_to_check) != 0:
            for c in list_to_check:
                # search in the chunks
                list_words = str(c).split(" ")
                for lc in list_words:
                    if lc not in CONNECTION_WORDS and lc in prompt_list:
                        word_to_replace = str(lc)
                        return True, word_to_replace

    # The function checks if the ners or chunks in the prompt containing part of the given words to replace
    def crop_word_from_input_replace(self, word_to_replace):
        """
        Analyzes the word to be replaced and extracts potential replacement candidates from the prompt.

        Args:
            word_to_replace (str): The word to be replaced.

        Returns:
            tuple: A tuple containing a boolean value indicating whether potential replacement words were found,
                   and the replacement word. Returns (False, None) if no suitable replacements are found.
        """
        # Split the prompt into list
        prompt_list = self.get_prompt().split(" ")
        prompt_list = list(map(str.lower, prompt_list))

        chunks_from_replace = self.nlp_engine.extract_nouns(word_to_replace)
        ners_from_replace = self.nlp_engine.extract_ners(word_to_replace)
        adjs_from_replace = self.nlp_engine.extract_ners(word_to_replace)
        if len(ners_from_replace) == 0 and len(chunks_from_replace) == 0 and len(adjs_from_replace) == 0:
            return False, None

        check_word, word_to_replace = self.check_in_list_to_replace(chunks_from_replace, prompt_list)
        if check_word:
            return True, word_to_replace

        check_word, word_to_replace = self.check_in_list_to_replace(ners_from_replace, prompt_list)
        if check_word:
            return True, word_to_replace

        check_word, word_to_replace = self.check_in_list_to_replace(adjs_from_replace, prompt_list)
        if check_word:
            return True, word_to_replace

    def check_valid_change(self, word_to_replace_with):
        """
        Analyzes the word to be replaced and attempts to extract suitable replacement candidates from the prompt.

        Args:
            word_to_replace (str): The word to be replaced.

        Returns:
            tuple: A tuple containing a boolean value indicating whether a suitable replacement word was found,
                   and the replacement word. Returns (False, None) if no suitable replacement is found.
        """
        chunks_from_replace = self.nlp_engine.extract_nouns(word_to_replace_with)
        ners_from_replace = self.nlp_engine.extract_ners(word_to_replace_with)
        adjs_from_replace = self.nlp_engine.extract_adjs(word_to_replace_with)
        if len(ners_from_replace) != 0:
            for w in ners_from_replace:
                if str(w) not in CONNECTION_WORDS:
                    return str(ners_from_replace[-1]), True

        if len(chunks_from_replace) != 0:
            for w in chunks_from_replace:
                if str(w) not in CONNECTION_WORDS:
                    return str(chunks_from_replace[-1]), True

        if len(adjs_from_replace) != 0:
            for w in adjs_from_replace:
                if str(w) not in CONNECTION_WORDS:
                    return str(adjs_from_replace[-1]), True

        return None, False

    def replace_text(self):
        """
        Allows the user to replace a word in the current prompt with another word of their choice.

        This method guides the user through the process of selecting a word to replace,
        confirming the choice, specifying a replacement word, and confirming the replacement.

        Note:
            This method updates the internal prompt of the Story_Teller instance.

        Returns:
            None
        """
        # Split the prompt into list
        prompt_list = self.get_prompt().split(" ")
        prompt_list = list(map(str.lower, prompt_list))

        word_to_replace = (self.nlp_engine.speak_input("What word do you want to replace?\n")).lower()

        bool_find_word, word_to_replace = self.crop_word_from_input_replace(word_to_replace)

        if not bool_find_word:
            prompt_to_say = "Can not change this word\n"
            self.nlp_engine.speak_print(prompt_to_say)
            return

        prompt_to_ask = "You chose the word " + word_to_replace + "\nIs it the right word?\n"
        check_word_correct = (self.nlp_engine.speak_input(prompt_to_ask)).lower()
        while check_word_correct == "no" or check_word_correct == "n" or check_word_correct == "false":
            word_to_replace = (self.nlp_engine.speak_input("What word do you want to replace?\n")).lower()
            # NEED TO EXTRACT HERE USING THE INPUT TESTING AND CROPPING
            prompt_to_ask = "You chose the word " + word_to_replace + "\nis it the right word?\n"
            check_word_correct = (self.nlp_engine.speak_input(prompt_to_ask)).lower()

        if check_word_correct == "yes" or check_word_correct == "y" or check_word_correct == "correct":
            if word_to_replace in prompt_list:
                index_to_replace = prompt_list.index(word_to_replace)
                if word_to_replace in CONNECTION_WORDS:
                    prompt_to_say = "Can not change this word\n"
                    self.nlp_engine.speak_print(prompt_to_say)
                    return
            # checks if the word exists in the current prompt
            else:
                prompt_to_say = "There is no such word  " + word_to_replace + " in the prompt\n"
                (self.nlp_engine.speak_print(prompt_to_say))
                prompt_to_say = "ok, the prompt is: " + self.get_prompt() + "\n"
                (self.nlp_engine.speak_print(prompt_to_say))
                return
            prompt_to_ask = "What word would you like to replace it with?\n"
            word_to_replace_with = (self.nlp_engine.speak_input(prompt_to_ask)).lower()
            word_to_replace_with, word_valid = self.check_valid_change(word_to_replace_with)
            print("The word to replace it with is: " + str(word_to_replace_with))
            if not word_valid:
                prompt_to_say = "Can not change this word\n"
                self.nlp_engine.speak_print(prompt_to_say)
                return
            prompt_list[index_to_replace] = word_to_replace_with
            new_prompt = str(" ".join(prompt_list))

        prompt_to_ask = "The new sentence is: " + str(new_prompt) + "\n" + "Are you satisfied with the result?\n"
        satisfy_yes_no = (self.nlp_engine.speak_input(prompt_to_ask)).lower()
        if satisfy_yes_no == "yes" or satisfy_yes_no == "y" or satisfy_yes_no == "correct":
            self.set_prompt(str(new_prompt))
        prompt_to_say = "Ok, the prompt is: " + self.get_prompt() + "\n"
        (self.nlp_engine.speak_print(prompt_to_say))
