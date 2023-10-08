

import spacy
import openai
from Hard_coded import stable_key
import requests
import json
from time import sleep
import en_core_web_sm
import random

from sentence_transformers import SentenceTransformer, util
from transformers import (
    pipeline,
    # set_seed,
    AutoTokenizer,
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    TopKLogitsWarper,
    TemperatureLogitsWarper,
    StoppingCriteriaList,
    MaxLengthCriteria, AutoModelForQuestionAnswering,
    AutoModelForCausalLM,
    BertForMaskedLM,
    # T5Tokenizer, T5ForConditionalGeneration
)
import torch
from happytransformer import HappyTextToText, TTSettings
#
#
# class GAN_eng:
#     def __init__(self):
#         self.gan_eng = diffusion.load_model("512x512_diffusion_uncond_finetune_008100.pt")
#         # Set the device to GPU if available
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         # Set the number of steps and the step size
#         self.num_steps = 1000
#         self.step_size = 0.01
#         # Define the image size
#         self.image_size = 512
#         # Define the transform pipeline
#         self.transform = Compose([Resize(self.image_size), ToPILImage()])
#
#     def create_painting(self, sentence):
#
#         # Define the text prompt
#         text_prompt = sentence
#
#         # Generate the image from the text prompt
#         image = diffusion.generate_from_text(model=self.gan_eng,
#                                              device=self.device,
#                                              text_prompt=text_prompt,
#                                              num_steps=self.num_steps,
#                                              step_size=self.step_size,
#                                              transform=self.transform)
#
#         # Save the image to a file
#         image.save("painting.png")


class Grammar_Engine:
    """
        Grammar Correction Engine
        This class handles grammar correction using a T5 model.
    """
    def __init__(self):
        self.grammar_model = HappyTextToText("T5", "vennify/t5-base-grammar-correction")
        self.args = TTSettings(num_beams=5, min_length=1)
    def correct_grammar(self, sentence):
        """
        Corrects the grammar of a given sentence.

        Parameters:
            sentence (str): The input sentence with possible grammar issues.

        Returns:
            str: The corrected sentence after grammar correction.
        """
        input_sentence = "grammar: " + sentence
        result = self.grammar_model.generate_text(input_sentence, args=self.args)
        return result.text

class Sentence_Similarity_Engine:
    """
    Sentence Similarity Engine
    This class provides functionality to compute the similarity between two sentences
    using pre-trained Sentence Transformer models.

    Attributes:
        sentence_model (SentenceTransformer): A pre-trained Sentence Transformer model.

    Methods:
        __init__: Initialize the Sentence Similarity Engine with a pre-trained model.
        get_similarity: Calculate the cosine similarity between two sentences.
    """
    def __init__(self):
        self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    def get_similarity(self, sentence1, sentence2):
        """
        Calculate the cosine similarity between two sentences.

        Parameters:
            sentence1 (str): The first sentence.
            sentence2 (str): The second sentence.

        Returns:
            float: The cosine similarity score between the two sentences.
        """
        embedding_1 = self.sentence_model.encode(sentence1, convert_to_tensor=True)
        embedding_2 = self.sentence_model.encode(sentence2, convert_to_tensor=True)
        return util.pytorch_cos_sim(embedding_1, embedding_2)

# self.QA_eng
class GPT_Engine:
    """
    GPT Engine
    This class provides functionality to interact with the GPT-2 model for text generation.

    Attributes:
        tokenizer (AutoTokenizer): A pre-trained GPT-2 tokenizer for tokenizing input text.
        model (AutoModelForCausalLM): A pre-trained GPT-2 model for text generation.
        logits_processor (LogitsProcessorList): A list of logits processors for post-processing model outputs.
        logits_warper (LogitsProcessorList): A list of logits warpers for modifying model output logits.

    Methods:
        __init__: Initialize the GPT Engine with a pre-trained tokenizer and model.
        extend_sentence: Generate a continuation of the input sentence using the GPT-2 model.
    """

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        self.model.config.pad_token_id = self.model.config.eos_token_id
        self.model.generation_config.pad_token_id = self.model.config.eos_token_id
        self.logits_processor = LogitsProcessorList([
            MinLengthLogitsProcessor(15, eos_token_id=self.model.config.eos_token_id),
        ])
        self.logits_warper = LogitsProcessorList([
            TopKLogitsWarper(50),
            TemperatureLogitsWarper(0.7),
        ])

        torch.manual_seed(0)

class MASK_Engine:
    """
      MASK Engine
      This class provides functionality to generate word predictions for masked tokens in a sentence using BERT models.

      Attributes:
          tokenizer (AutoTokenizer): A pre-trained BERT tokenizer for tokenizing input text.
          model (BertForMaskedLM): A pre-trained BERT model for masked language modeling.

      Methods:
          __init__: Initialize the MASK Engine with a pre-trained tokenizer and model.
          mask_extend: Replace [MASK] tokens in the input sentence with predicted words from the BERT model.
      """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-large-cased-whole-word-masking")
        self.model = BertForMaskedLM.from_pretrained("bert-large-cased-whole-word-masking")

    def mask_extend(self, sentence):
        """
        Replace [MASK] tokens in the input sentence with predicted words from the BERT model.

        Parameters:
            sentence (str): The input sentence with [MASK] tokens.

        Returns:
            str: The sentence with [MASK] tokens replaced by predicted words.
        """
        inputs = self.tokenizer(sentence, return_tensors="pt")

        with torch.no_grad():
            logits = self.model(**inputs).logits

        # retrieve index of [MASK]
        mask_token_index = \
            (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]

        predicted_token_ids = logits[0, mask_token_index].topk(k=5, dim=-1).indices
        predicted_token_ids = predicted_token_ids.tolist()
        predicted_token_ids = [item for sublist in predicted_token_ids for item in sublist]
        random.shuffle(predicted_token_ids)

        mask_answers = [self.tokenizer.decode(token_id) for token_id in predicted_token_ids]

        # replace [MASK] with a randomly chosen predicted answer
        random_answer = random.choice(mask_answers)
        if random_answer.lower() in sentence.lower():
            mask_answers.remove(random_answer)
            random_answer = random.choice(mask_answers)
        return sentence.replace("[MASK]", random_answer)


class QA_Engine:
    """
        QA Engine
        This class provides functionality for performing question answering using a pre-trained DistilBERT model.

        Attributes:
            tokenizer (AutoTokenizer): A pre-trained tokenizer for tokenizing input text.
            model (AutoModelForQuestionAnswering): A pre-trained DistilBERT model for question answering.
            device (str): The device to run the model on (CPU or GPU).

        Methods:
            __init__: Initialize the QA Engine with a pre-trained tokenizer and model.
            QA_extract: Perform question answering on the given context using a question.
        """
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased-distilled-squad")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def QA_extract(self, question, context):
        """
                Perform question answering on the given context using a question.

                Parameters:
                    question (str): The question to ask about the context.
                    context (str): The text containing the information to extract the answer from.

                Returns:
                    str: The answer to the question extracted from the context.
                """
        # Tokenize the input text and question
        inputs = self.tokenizer(question, context, add_special_tokens=True, return_tensors="pt")

        # Use the model to predict the answer
        start_scores, end_scores = self.model(**inputs).start_logits, self.model(
            **inputs).end_logits
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores) + 1
        answer_tokens = inputs["input_ids"][0][start_index:end_index]
        answer = self.tokenizer.decode(answer_tokens)
        if "[SEP]" in answer:
            answer = answer.replace("[SEP]" , "")
        return answer

class SpellCheckEngine:
    """
    Spell Check Engine
    This class provides functionality for performing spell checking using a pre-trained text-to-text model.

    Attributes:
        model (Pipeline): A pre-trained text-to-text model for spell checking.

    Methods:
        __init__: Initialize the SpellCheckEngine with a pre-trained model.
        spell_check: Perform spell checking on the given sentence.

    Note:
        This class requires the 'happytransformer' library to be installed.
    """
    def __init__(self):
        self.model = pipeline("text2text-generation", model="oliverguhr/spelling-correction-english-base")

    def spell_check(self, sentence):
        """
        Perform spell checking on the given sentence.

        Parameters:
            sentence (str): The input sentence to be spell checked.

        Returns:
            str: The spell-checked version of the input sentence.
        """
        return self.model(sentence, max_length=2048)[0]['generated_text']


class EnhanceEngine:
    """
        Enhance Engine
        This class provides functionality for enhancing sentences using a masking-based approach.

        Attributes:
            mask_engine (MASK_Engine): An instance of the MASK_Engine class for masking words.

        Methods:
            __init__: Initialize the EnhanceEngine with a MASK_Engine instance.
            enhance_sentence: Enhance the given sentence using masking.

        Note:
            This class relies on the MASK_Engine class to perform word masking and enhancement.
        """
    def __init__(self):
        self.mask_engine = MASK_Engine()

    def enhance_sentence(self, sentence):
        """
                Enhance the given sentence using masking-based approach.

                Parameters:
                    sentence (str): The input sentence to be enhanced.

                Returns:
                    str: The enhanced version of the input sentence.
                """
        enhanced = sentence.replace(".", "")
        for i in range(2):
            masked_input = enhanced + " [MASK]."
            enhanced = self.mask_engine.mask_extend(masked_input)
            enhanced = enhanced.strip(".")
        # remove the '#' enhance
        new_sentence_list = enhanced.split(" ")
        new_temp = new_sentence_list
        for s in new_sentence_list:
            if '#' in s:
                new_temp.remove(s)
        if '#' in new_temp[-1]:
            new_temp.remove(new_temp[-1])
        enhanced = " ".join(new_temp)
        return enhanced


class NLP_Engine:
    """
        Natural Language Processing Engine
        This class provides various natural language processing capabilities, including grammar correction,
        question answering, text enhancement, similarity measurement, and more.

        Attributes:
            socket (object): The socket object used for communication in a specific environment.
            __developer_env (bool): A flag indicating whether the environment is a developer environment.
            __spacy_engine (spacy.Language): A Spacy NLP language model instance for text processing.
            grammar_engine (Grammar_Engine): An instance of the Grammar_Engine class for grammar correction.
            gpt_engine (GPT_Engine): An instance of the GPT_Engine class for text generation.
            qa_engine (QA_Engine): An instance of the QA_Engine class for question answering.
            mask_engine (MASK_Engine): An instance of the MASK_Engine class for word masking.
            enhance_engine (EnhanceEngine): An instance of the EnhanceEngine class for text enhancement.
            spell_checker (SpellCheckEngine): An instance of the SpellCheckEngine class for spelling correction.
            similarity_engine (Sentence_Similarity_Engine): An instance of the Sentence_Similarity_Engine class
                                                           for measuring sentence similarity.
    """
    def __init__(self, developer_env=False, socketio=None):
        self.socket = socketio
        self.__developer_env = developer_env
        nlp = spacy.load('en_core_web_sm')
        # nlp.add_pipe("merge_entities")
        # nlp.add_pipe("merge_noun_chunks")
        self.__spacy_engine = nlp

        # self.__speak_engine = pyttsx3.init('sapi5')
        # voice = self.__speak_engine.getProperty('voices')
        # self.__speak_engine.setProperty('voice', voice[1].id)
        # self.__speak_engine.setProperty('rate', 160)
        self.grammar_engine = Grammar_Engine()
        self.gpt_engine = GPT_Engine()
        self.qa_engine = QA_Engine()
        self.mask_engine = MASK_Engine()
        self.enhance_engine = EnhanceEngine()
        self.spell_checker = SpellCheckEngine()
        self.similarity_engine = Sentence_Similarity_Engine()

    def extend_sentence_with_MASK(self, sentence):
        """
                Extends the sentence by replacing nouns with masked versions and randomly predicting words.

                This method replaces each noun in the sentence with a masked version of the noun and then
                uses the MASK_Engine to predict words to complete the sentence.

                Parameters:
                    sentence (str): The input sentence to be extended.

                Returns:
                    str: The extended sentence with masked nouns and predicted words.

                Example:
                    nlp_engine = NLP_Engine()
                    sentence = "The cat is sitting on the mat."
                    extended_sentence = nlp_engine.extend_sentence_with_MASK(sentence)
                """
        nouns = self.extract_nouns(sentence)
        for noun in nouns:
            sentence = sentence.replace(noun, f'[MASK] {noun}')
            sentence = self.mask_engine.mask_extend(sentence)
        return sentence

    def text_to_img_for_web_dale2(self, prompt):
        """
            Generates images based on the provided text prompt using the OpenAI API.

            This method sends a text prompt to the OpenAI API to generate an image based on the provided prompt.
            It uses the "512x512_diffusion_uncond_finetune_008100.pt" model to generate images.

            Parameters:
                prompt (str): The text prompt for generating the image.

            Returns:
                list: A list containing URLs of the generated images.

            Example:
                nlp_engine = NLP_Engine()
                prompt = "A beautiful sunset over the mountains."
                image_urls = nlp_engine.text_to_img_for_web_dale2(prompt)
            """
        try:
            file_path = 'my_api_key.txt'
            with open(file_path, 'r') as file:
                openai.api_key = file.read()
            res = openai.Image.create(
                prompt="512x512 size img - " + prompt,
                n=1,
                size="512x512"
            )
            return [res.data[0].url]
        except Exception:
            return ["API KEY - Something Went Wrong"]
            # return os.environ.keys()


    def text_to_img_for_web(self, prompt):
        """
        Generates images based on the provided text prompt using the Stable Diffusion API.

        This method sends a text prompt to the Stable Diffusion API to generate an image based on the provided prompt.
        It provides additional parameters for image generation such as negative prompts, image dimensions, and more.

        Parameters:
            prompt (str): The text prompt for generating the image.

        Returns:
            list: A list containing URLs of the generated images.

        Example:
            nlp_engine = NLP_Engine()
            prompt = "A beautiful sunset over the mountains."
            image_urls = nlp_engine.text_to_img_for_web(prompt)
        """
        url = "https://stablediffusionapi.com/api/v3/text2img"
        payload = {
            "key": stable_key,
            "prompt": prompt,
            "negative_prompt": "((out of frame)), ((extra fingers)), mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), (((tiling))), ((naked)), ((tile)), ((fleshpile)), ((ugly)), (((abstract))), blurry, ((bad anatomy)), ((bad proportions)), ((extra limbs)), cloned face, (((skinny))), glitchy, ((extra breasts)), ((double torso)), ((extra arms)), ((extra hands)), ((mangled fingers)), ((missing breasts)), (missing lips), ((ugly face)), ((fat)), ((extra legs)), anime",
            "width": "512",
            "height": "512",
            "samples": "1",
            "num_inference_steps": "20",
            "seed": 'null',
            "guidance_scale": 7.5,
            "webhook": 'null',
            "track_id": 'null'
        }
        response = requests.post(url, json=payload)
        # check if the response was okay but the monthly limit was exceeded and print accordingly
        decoded_response = json.loads(response.content)
        if decoded_response["status"] == "error" and "exceeded" in decoded_response["message"]:
            return ["The monthly limit for the API was exceeded. Please try again later."]
        if response.status_code == 200:
            image_url = decoded_response.get('output')
            if not image_url:
                fetch_url = decoded_response.get('fetch_result')
                estimate_time = decoded_response.get('eta')
                print(f"Waiting {estimate_time} seconds for the image to be generated...")
                sleep(estimate_time + 2)
                response = requests.post(fetch_url, json=payload)
                image_url = json.loads(response.content).get('output')
            return image_url
          
    def get_spacy(self):
        return self.__spacy_engine

      
    def extend_sentence(self, sentence):
        """
        Extends a given sentence using the GPT-2 language model.

        This method takes a sentence as input and generates an extended version using the GPT-2 language model.
        It utilizes the model to sample additional text based on the input and stopping criteria.

        Parameters:
            sentence (str): The input sentence to be extended.

        Returns:
            str: An extended version of the input sentence.

        Example:
            nlp_engine = NLP_Engine()
            input_sentence = "The sun was setting over the horizon."
            extended_sentence = nlp_engine.extend_sentence(input_sentence)
        """
        input_prompt = sentence
        input_ids = self.gpt_engine.tokenizer(input_prompt, return_tensors="pt").input_ids
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=len(sentence) + 2)])
        outputs = self.gpt_engine.model.sample(
            input_ids,
            logits_processor=self.gpt_engine.logits_processor,
            logits_warper=self.gpt_engine.logits_warper,
            stopping_criteria=stopping_criteria,
        )

        output_prompt = self.gpt_engine.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        # cropping the sentence, so it won't stop in a middle of a sentence
        return ".".join((output_prompt[0].split("."))[:-1]) + "."


    def extract_ners_tokens(self, sentence, ner_type=None):
        """
                Extracts named entities as tokens from a sentence.

                This method processes a sentence and extracts named entities based on the specified entity type.
                If no entity type is provided, all named entities found in the sentence are returned as tokens.

                Parameters:
                    sentence (str): The input sentence to extract named entities from.
                    ner_type (str, optional): The specific entity type to extract (e.g., "PERSON", "ORG").
                                              If None, all named entities are extracted.

                Returns:
                    list: A list of named entity tokens extracted from the sentence.

                Example:
                    nlp_engine = NLP_Engine()
                    input_sentence = "Apple Inc. was founded by Steve Jobs."
                    person_entities = nlp_engine.extract_ners_tokens(input_sentence, ner_type="PERSON")
                    all_entities = nlp_engine.extract_ners_tokens(input_sentence)
                """
        doc = self.__spacy_engine(sentence)
        if ner_type == None:
            relevant_ents = [ent for ent in doc.ents]
        else:
            relevant_ents = [ent for ent in doc.ents if ent.label_ == ner_type]
        return relevant_ents

    def extract_ners(self, sentence, ner_type=None):
        """
        Extracts named entities from a sentence.

        This method processes a sentence and extracts named entities based on the specified entity type.
        If no entity type is provided, named entities of various types (e.g., PERSON, ORG, GPE) are extracted.

        Parameters:
            sentence (str): The input sentence to extract named entities from.
            ner_type (str, optional): The specific entity type to extract (e.g., "PERSON", "ORG").
                                      If None, named entities of various types are extracted.

        Returns:
            list: A list of named entities extracted from the sentence.

        Example:
            nlp_engine = NLP_Engine()
            input_sentence = "Apple Inc. was founded by Steve Jobs in Cupertino."
            person_entities = nlp_engine.extract_ners(input_sentence, ner_type="PERSON")
            location_entities = nlp_engine.extract_ners(input_sentence, ner_type="GPE")
            all_entities = nlp_engine.extract_ners(input_sentence)
        """
        # person - People, including fictional.
        # FAC - Buildings, airports, highways, bridges, etc.
        # ORG - Companies, agencies, institutions, etc.
        # GPE - Countries, cities, states.
        # LOC - Non-GPE locations, mountain ranges, bodies of water.
        # PRODUCT - Objects, vehicles, foods, etc. (Not services.)
        ners_labels = ["PERSON",  "FAC", "ORG", "GPE", "LOC", "PRODUCT"]
        doc = self.__spacy_engine(sentence)
        if ner_type is None:
            relevant_ents = [ent.text for ent in doc.ents if ent.label_ in ners_labels]
        else:
            relevant_ents = [ent.text for ent in doc.ents if ent.label_ == ner_type]
        return relevant_ents

    def extract_adjs(self, sentence):
        """
        Extracts adjectives from a sentence.

        This method processes a sentence and extracts adjectives present in the sentence.

        Parameters:
            sentence (str): The input sentence to extract adjectives from.

        Returns:
            list: A list of adjectives extracted from the sentence.

        Example:
            nlp_engine = NLP_Engine()
            input_sentence = "The quick brown fox jumps over the lazy dog."
            adjectives = nlp_engine.extract_adjs(input_sentence)
        """
        doc = self.__spacy_engine(sentence)
        return [token.text for token in doc if token.pos_ == "ADJ"]


    def extract_adjs_with_first_ancestor(self, sentence, ancestor):
        """
            Extract Adjectives with a Specific First Ancestor

            This function takes a sentence and a target ancestor word as input and extracts all adjectives within the sentence
            that have the specified ancestor as their immediate head in the dependency tree.

            Parameters:
            ------------
            self : Object
                An instance of the class containing this method.

            sentence : str
                The input sentence from which adjectives need to be extracted.

            ancestor : str
                The target ancestor word. Adjectives that are directly dependent on this word in the dependency tree will be extracted.

            Returns:
            -----------
            list
                A list of adjectives that have the provided 'ancestor' word as their immediate head in the dependency tree.

            Notes:
            --------
            - The function utilizes the Spacy NLP engine provided within the class instance to process the input sentence.
            - It searches for adjective tokens within the processed sentence.
            - For each adjective token found, the function checks if its immediate head matches the provided 'ancestor' word.
            - Adjectives meeting the criteria are collected in a list and returned.

            Example:
            ---------
            Consider the following function call:
            instance = YourClass()
            sentence = "The big red apple is delicious."
            ancestor_word = "apple"
            result = instance.extract_adjs_with_first_ancestor(sentence, ancestor_word)

            In this case, the function will return:
            ['big', 'red']
            These are the adjectives "big" and "red" which are directly dependent on the word "apple" in the given sentence.

            """
        doc = self.__spacy_engine(sentence)
        ancestor_dict = {}
        for token in doc:
            if token.pos_ == "ADJ":
                ancestor_dict[token.text] = token.head.text
        return [adj for adj, head in ancestor_dict.items() if head == ancestor]

    def extract_noun_tokens(self, sentence):
        """
           Extract root noun tokens from a sentence.

           Parameters:
           ------------
           sentence : str
               The input sentence.

           Returns:
           -----------
           list
               List of root noun tokens extracted from the sentence.
           """
        doc = self.__spacy_engine(sentence)
        return [chunk.root for chunk in doc.noun_chunks]

    def extract_nouns(self, sentence):
        """
            Extract nouns from a sentence's noun chunks.

            Parameters:
            ------------
            sentence : str
                The input sentence.

            Returns:
            -----------
            list
                List of strings representing the text of root nouns within noun chunks.
            """
        doc = self.__spacy_engine(sentence)
        return [chunk.root.text for chunk in doc.noun_chunks]

    def extract_noun_chunks(self, sentence):
        """
            Extract root noun chunks from a sentence.

            Parameters:
            ------------
            sentence : str
                The input sentence.

            Returns:
            -----------
            list
                List of strings representing the text nouns chunks.
                    """
        doc = self.__spacy_engine(sentence)
        return [chunk for chunk in doc.noun_chunks]


    def get_semantic_tree(self, target_noun, sentence):
        """
            Get the subtree rooted at the target noun in a sentence.

            Parameters:
            ------------
            target_noun : str
                The target noun for which the semantic subtree needs to be extracted.

            sentence : str
                The input sentence.

            Returns:
            -----------
            Span or None
                The subtree rooted at the target noun as a Spacy Span object, or None if the target noun is not found.
            """
        doc = self.__spacy_engine(sentence)
        for token in doc:
            if token.text == target_noun:
                return token.subtree
        return None

    def get_target_noun_tree(self, target_noun, sentence):
        """
            Get the target noun tree within a sentence.

            This function identifies the noun chunk containing the target noun and constructs a subtree that includes the target noun,
            any prepositions following it, and any subsequent noun chunks.

            Parameters:
            ------------
            target_noun : str
                The target noun for which the tree needs to be extracted.

            sentence : str
                The input sentence.

            Returns:
            -----------
            str or None
                A string representing the target noun tree, or None if the target noun is not found.

            Notes:
            --------
            - The function utilizes the Spacy NLP engine provided within the class instance to process the input sentence.
            - It identifies noun chunks in the sentence and checks if the target noun is part of any of the noun chunks.
            - If the target noun is found, the function constructs a subtree that includes the target noun, prepositions, and subsequent nouns.
            - The constructed subtree is returned as a string.
            - If the target noun is not found, the function returns None.

            Example:
            ---------
            Consider the following function call:
            instance = YourClass()
            sentence = "The cat on the mat is sleeping."
            target = "mat"
            result = instance.get_target_noun_tree(target, sentence)

            In this case, the function will return:
            "the mat is sleeping"
            This corresponds to the noun chunk containing "mat" and its related words in the sentence.

            """
        doc = self.__spacy_engine(sentence)
        noun_chunks = [n for n in doc.noun_chunks]
        for index in range(len(noun_chunks)):
            if target_noun in noun_chunks[index].text:
                target_noun_chunks = noun_chunks[index].text
                adp = [n.text for n in noun_chunks[index].rights if n.pos_ == 'ADP']
                if len(adp) <= 1:
                    return target_noun_chunks + " " + adp[0] + " " + noun_chunks[index + 1].text
                return target_noun_chunks
        return None

    def extract_subject(self, sentence):
        """
           Extract the subject of a sentence.

           This function identifies the subject of the given sentence and returns it as a Spacy Doc object.

           Parameters:
           ------------
           sentence : str
               The input sentence.

           Returns:
           -----------
           Doc or str
               A Spacy Doc object representing the subject of the sentence, or "no noun subject" if no subject is found.

           Notes:
           --------
           - The function utilizes the Spacy NLP engine provided within the class instance to process the input sentence.
           - It iterates through the tokens in the processed sentence and looks for a token with a "subj" dependency.
           - Once a subject token is found, the function constructs a subtree rooted at the subject token and returns it as a Doc object.
           - If no subject is found, the function returns the string "no noun subject".

           Example:
           ---------
           Consider the following function call:
           instance = YourClass()
           sentence = "The cat is sleeping."
           result = instance.extract_subject(sentence)

           In this case, the function will return a Spacy Doc object representing the subject "The cat".

           """
        doc = self.__spacy_engine(sentence)
        for token in doc:
            if ("subj" in token.dep_):
                subtree = list(token.subtree)
                start = subtree[0].i
                end = subtree[-1].i + 1
                return doc[start:end]
        return "no noun subject"

    def check_same_POS(self, word1, word2, sentence1, sentence2):
        """
            Check if two words have the same part-of-speech (POS) tag in their respective sentences.

            This function compares the part-of-speech (POS) tags of two words in their respective sentences and
            returns True if they have the same POS tag, and False otherwise.

            Parameters:
            ------------
            word1 : str
                The first word for comparison.

            word2 : str
                The second word for comparison.

            sentence1 : str
                The first sentence containing word1.

            sentence2 : str
                The second sentence containing word2.

            Returns:
            -----------
            bool
                True if word1 and word2 have the same POS tag, False otherwise.

            Notes:
            --------
            - The function utilizes the Spacy NLP engine provided within the class instance to process the input sentences.
            - It searches for word1 and word2 in their respective sentences and retrieves their POS tags.
            - The comparison is based on the equality of the POS tags.
            - Returns True if both words have the same POS tag, and False otherwise.

            Example:
            ---------
            Consider the following function call:
            instance = YourClass()
            word1 = "cat"
            word2 = "dog"
            sentence1 = "The black cat is sleeping."
            sentence2 = "The brown dog is barking."
            result = instance.check_same_POS(word1, word2, sentence1, sentence2)

            In this case, the function will return False, as "cat" and "dog" do not have the same POS tag.

            """
        doc1 = self.__spacy_engine(sentence1)
        doc2 = self.__spacy_engine(sentence2)
        pos1, pos2 = "", ""
        for tok in doc1:
            if tok.text == str(word1):
                pos1 = str(tok.pos_)
        for tok in doc2:
            if tok.text == str(word2):
                pos2 = str(tok.pos_)
        return pos1 == pos2


    def sentiment_analysis(self, sentence):
        pass

    # def speak_input(self, prompt1, prompt2=""):
    #     print(prompt1 + prompt2)
    #     self.socket.emit('output', prompt1 + prompt2)
    #     return ""
    #
    # def print_input(self, prompt1):
    #     print(prompt1)
    #     self.socket.emit('output', prompt1)
    #     return ""

    def speak_input(self, prompt1, prompt2=""):
        if self.__developer_env:
            return input(prompt1 + prompt2)
        print(prompt1 + prompt2)
        # self.__speak_engine.say(prompt1)
        # self.__speak_engine.runAndWait()
        return input()

    def speak_print(self, prompt):
        print(prompt)
        # if not self.__developer_env:
            # self.__speak_engine.say(prompt)
            # self.__speak_engine.runAndWait()


    def add_descriptions(self, sentence, word):
        """
            Add descriptions to a word in a sentence.

            This function modifies the given sentence to add descriptions to a specified word, making use of adjectives and prepositional phrases.

            Parameters:
            ------------
            sentence : str
                The input sentence.

            word : str
                The word to which descriptions are to be added.

            Returns:
            -----------
            str
                The modified sentence with added descriptions for the specified word.

            Notes:
            --------
            - The function utilizes the Spacy NLP engine provided within the class instance to process the input sentence.
            - If the specified word is not present in the sentence, it is added at the beginning of the sentence with a generic "is" clause.
            - Adjectives are extracted from the sentence using the extract_adjs() method.
            - The function then identifies prepositional phrases related to the specified word and constructs new phrases using adjectives and prepositions.
            - The descriptions are appended to the word, and the modified sentence is returned.

            Example:
            ---------
            Consider the following function call:
            instance = YourClass()
            sentence = "The cat is sleeping."
            word = "cat"
            result = instance.add_descriptions(sentence, word)

            In this case, the function might return a modified sentence:
            "The sleepy black and white cat under the tree is sleeping."
            The word "cat" is described using the adjectives "sleepy", "black", and "white", and a prepositional phrase "under the tree".

            """
        if word not in sentence:
            sentence = word + " " + " is " + " " + sentence
        adj = self.extract_adjs(sentence)
        doc = self.__spacy_engine(sentence)
        adps = []


        noun_chunks = doc.noun_chunks

        for token in doc:
            if token.pos_ == "ADP":
                children = token.children
                for child in children:
                    child_chunk = ""
                    for chunk in noun_chunks:
                        if child in chunk:
                            child_chunk = chunk
                    if child.dep_ == "pobj":
                        adps.append(token.text + " " + child_chunk.text)
        return "".join(adj) + " " +  word + " ".join(adps)

    def add_descriptions_QA(self, sentence, word):
        """
           Add descriptions using a question-answering approach.

           This function generates a question about the description of a specified word in a sentence,
           and then extracts information from the generated question using a question-answering engine.

           Parameters:
           ------------
           sentence : str
               The input sentence.

           word : str
               The word for which descriptions are sought.

           Returns:
           -----------
           str
               The modified answer that includes descriptions of the specified word.

           Notes:
           --------
           - The function generates a question of the form "what is the description of <word>?".
           - It uses a question-answering engine (self.qa_engine) to extract information from the question about the input sentence.
           - If the answer obtained from the QA engine contains the specified word, it is returned as the final answer.
           - If the answer does not contain the word, the function appends the word to the answer before returning.

           Example:
           ---------
           Consider the following function call:
           instance = YourClass()
           sentence = "The cat is sleeping."
           word = "cat"
           result = instance.add_descriptions_QA(sentence, word)

           In this case, the function might return a modified answer:
           "The sleepy cat under the tree is sleeping."
           The function generates a question about the description of "cat" and extracts the relevant information from the sentence.
           If the answer does not contain the word "cat", it is appended to the answer.

           """
        question = f'what is the description of {word}?'
        answer = self.qa_engine.QA_extract(question, sentence)
        if word in answer:
            return answer
        else:
            return f'{answer} {word}'

    def check_pos_correct(self, user_input, pos='ADJ'):
        """
            Check if a specific part-of-speech (POS) tag is present in the user input.

            This function analyzes the given user input using a Spacy NLP engine and determines if the specified POS tag is present.

            Parameters:
            ------------
            user_input : str
                The input text provided by the user.

            pos : str, optional
                The target POS tag to check for. Defaults to 'ADJ'.

            Returns:
            -----------
            bool
                True if the specified POS tag is found in the user input, False otherwise.
            """
        doc = self.__spacy_engine(user_input)
        for token in doc:
            if token.pos_ == pos:
                return True
        return False

    def correct_grammar(self, sentence):
        """
            Correct the grammar of a given sentence.

            This function utilizes a grammar correction engine to correct the grammar of the provided sentence.

            Parameters:
            ------------
            sentence : str
                The input sentence with potential grammar errors.

            Returns:
            -----------
            str
                The sentence with corrected grammar.

            Notes:
            --------
            - The function makes use of an external grammar correction engine (self.grammar_engine) to correct the input sentence.
            - It returns the input sentence after applying grammar corrections.
            """
        return self.grammar_engine.correct_grammar(sentence)

    def spell_check(self, sentence):
        """
           Perform spell-check on a given sentence.

           This function utilizes a spell-checking engine to correct spelling errors in the provided sentence.

           Parameters:
           ------------
           sentence : str
               The input sentence with potential spelling errors.

           Returns:
           -----------
           str
               The sentence with corrected spelling.

           Notes:
           --------
           - The function makes use of an external spell-checking engine (self.spell_checker) to correct the input sentence.
           - It returns the input sentence after applying spelling corrections.

           """
        return self.spell_checker.spell_check(sentence)

    def check_input(self, sentence):
        """
            Check and correct input sentence for spelling and grammar errors.

            This function performs both spelling and grammar checks on the input sentence, correcting errors if found.

            Parameters:
            ------------
            sentence : str
                The input sentence with potential spelling and grammar errors.

            Returns:
            -----------
            str
                The corrected sentence after performing spelling and grammar checks.

            Notes:
            --------
            - The function first applies spell-checking to the input sentence using the spell_check() method.
            - Then, it applies grammar correction using the correct_grammar() method to the spell-checked sentence.
            - The function returns the final sentence after both spelling and grammar checks have been applied.


            """
        spell_checked_input = self.spell_check(sentence)
        grammar_checked_input = self.correct_grammar(spell_checked_input)
        return grammar_checked_input

    def enhance_sentence(self, sentence):
        """
            Enhance a sentence by adding adjectives and improving it.

            This function enhances the input sentence by extending it with added adjectives using MASKing,
            then applying enhancements through an enhancement engine.
            It also calculates and logs the similarity between the original and enhanced sentences.

            Parameters:
            ------------
            sentence : str
                The input sentence to be enhanced.

            Returns:
            -----------
            str
                The enhanced sentence after applying improvements.

            Notes:
            --------
            - The function uses a method (extend_sentence_with_MASK) to add adjectives to the input sentence.
            - The enhanced sentence is then obtained using an enhancement engine (self.enhance_engine).
            - The function calculates the similarity between the original and enhanced sentences using a similarity engine.
            - The similarity value is printed and logged in a file named "similarity.txt".
            - The final enhanced sentence is returned.


            """
        sentence_with_added_adjs = self.extend_sentence_with_MASK(sentence)
        ret_sentence = self.enhance_engine.enhance_sentence(sentence_with_added_adjs)
        ret_sentence = ret_sentence.replace(".", "")
        Similarity = self.similarity_engine.get_similarity(sentence, ret_sentence)
        print(f'Similarity between the sentence and the enhancement: {Similarity}')
        with open("similarity.txt", "a") as f:
            f.write(f'{sentence}\n')
            f.write(f'{ret_sentence}\n')
            f.write(f'Similarity between the sentence and the enhancement: {Similarity}\n\n')
        return ret_sentence

    def QA_extract_user_descriptions(self, sentence):
        """
           Extract user descriptions using a question-answering approach.

           This function generates a question about the user's descriptions in the provided sentence and extracts information
           from the generated question using a question-answering engine.

           Parameters:
           ------------
           sentence : str
               The input sentence containing user descriptions.

           Returns:
           -----------
           str
               The extracted user descriptions.

           Notes:
           --------
           - The function generates a question of the form "Painting of what?" to extract user descriptions from the sentence.
           - It uses a question-answering engine (self.qa_engine) to extract information from the question about the input sentence.

           """
        question = f'Painting of what?'
        return self.qa_engine.QA_extract(question, sentence)
