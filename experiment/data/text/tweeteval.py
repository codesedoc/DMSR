import codecsimport copyimport jsonimport pickleimport randomimport sysfrom abc import ABCimport datasetsimport nltkimport numpy as npfrom datasets import Datasetfrom nltk import WordNetLemmatizerfrom experiment.utils import read_lines_from_file, read_from_csv_file, dump_info_to_jsonfrom src.nlps.data import Data, data_register, DatasetSplitType, TaskType, GeneralDataset, DataContainer, \    DataDirCategoryfrom typing import Dict, Tuple, Union, Any, MutableMapping, List, Iterableimport osimport pandas as pdfrom src.nlps.data.data import ALL_DATASET_SPLITfrom src.nlps.utils.utils import max_length_of_sequencesfrom nltk.corpus import wordsfrom nltk.corpus import wordnet2021from nltk.stem import WordNetLemmatizerclass TweetEval(Data, ABC):    _abbreviation = 'tweeteval'    _metric_name_path = 'accuracy'    _task_type = TaskType.CLASSIFICATION    def __init__(self, *args, **kwargs):        super().__init__(*args, **kwargs)        self._max_length = None    @property    def max_length(self):        if not isinstance(self._max_length, int):            self._max_length = max_length_of_sequences(self, dataset=list(self._dataset.values()))        return self._max_length@data_registerclass TESentiment(TweetEval):    _abbreviation = os.path.join(TweetEval._abbreviation, 'sentiment')    def __init__(self, *args, **kwargs):        super(TESentiment, self).__init__(*args, **kwargs)        # nltk.download('words')        # nltk.download('wordnet')        # nltk.download('wordnet2021')        # self._word_library = set(words.words() + list(wordnet2021.words()))        # self._wordnet_lemmatizer = WordNetLemmatizer()        self._invalid_words = {"http"}    def _wash_word(self, word: Union[str, Iterable[str]]):        if isinstance(word, str):            for iw in self._invalid_words:                if iw in word:                    return True            return False            # if "'" in word:            #     word_peaces = word.split("'")            #     result = self._wash_word(word_peaces)            #     return np.array(result, dtype=np.bool_).any()            # else:            #     pass                # lemma = self._wordnet_lemmatizer.lemmatize(word).lower()                # return lemma not in self._word_library        elif isinstance(word, Iterable):            return [self._wash_word(w) for w in word]        else:            raise ValueError    def _wash_sentence(self, sentence:Union[str, Iterable[str]]):        if isinstance(sentence, str):            sentence = sentence.strip('"').strip("")            word_list = sentence.split(" ")            bad_words = self._wash_word(word_list)            new_words = []            remove_recode = []            for i, w, b in zip(range(len(word_list)), word_list, bad_words):                if b:                    remove_recode.append((i, w))                    continue                new_words.append(w)            new_sentence = " ".join(new_words)            return new_sentence, remove_recode        elif isinstance(sentence, Iterable):            result = []            modified_info = []            for s in sentence:                ns, mi = self._wash_sentence(s)                result.append(ns)                modified_info.append(mi)            return result, modified_info        else:            raise ValueError    def _preprocess(self):            splits2name: Dict[str, DatasetSplitType] = {                DatasetSplitType.TRAIN: 'train',                DatasetSplitType.TEST: 'test',                DatasetSplitType.VALIDATION: 'val'            }            extensions = 'txt'            for s, n in splits2name.items():                if not s in self._preprocessed_files:                    continue                output_path = self._preprocessed_files[s]                label_raw_path = os.path.join(self.raw_dir, f'{n}_labels.{extensions}')                text_raw_path = os.path.join(self.raw_dir, f'{n}_text.{extensions}')                # mapping_path = os.path.join(self.raw_dir,f"mapping.{extensions}")                raw_path = label_raw_path, text_raw_path                if not np.array([os.path.isfile(rp) for rp in raw_path], dtype=np.bool_).all():                    raise ValueError                texts = read_lines_from_file(text_raw_path)                if s != DatasetSplitType.TEST:                    for t_i, t in enumerate(texts):                        try:                            texts[t_i] = codecs.decode(t, encoding="unicode_escape")                        except:                            print(f"Can not decode this sentence in 'unicode_escape' set.\t {t}")                labels = [int(v) for v in read_lines_from_file(label_raw_path)]                washed_texts, modified_info = self._wash_sentence(texts)                if np.array([len(mi) for mi in modified_info], dtype=np.bool_).any():                    washed_examples = []                    for i, mi in enumerate(modified_info):                        if len(mi) > 0:                            washed_examples.append({                                "order": i,                                "original_sentence": texts[i],                                "washed_result": washed_texts[i],                                "removed_words": [wi[-1] for wi in mi]                            })                    washed_info_path = f"{output_path}_washed.json"                    json_data = {                        "Data Source": self.abbreviation,                        "Number of Examples": len(labels),                        "Number of Washed Examples": len(washed_examples),                        "Washing Record": washed_examples                    }                    dump_info_to_json(information=json_data, file_path=washed_info_path, indent=4, ensure_ascii=False)                dataset = Dataset.from_dict({                            "sentence": texts,                            "label": labels                        })                dataset.to_csv(output_path, index=False)    @property    def input_column_name(self):        return "sentence"    @property    def label_column_name(self):        return "label"    @property    def an_sample(self) -> Tuple[Any]:        return "I am happy", 2    def _class_names(self) -> List[str]:        return ["positive", "negative", "neutral"]if __name__ == "__main__":    sys.argv.extend(['--dataset', 'tweeteval/sentiment', '--force_preprocess'])    data = TESentiment()    dataset = data.dataset()    for d in dataset:        d.to_csv("tmp/dataset.csv")    pass