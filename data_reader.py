import xml.etree.ElementTree as ET
import re
import csv
import os

import torch
from torchtext import data, vocab
import spacy
import dill

class FileNames:
    "Stores all file names and paths to data"
    def __init__(self, year):
        self.reviewtype = 'restaurants'
        self.year = year
        self.base_file_name = 'ABSA' + str(year) + '_' + self.reviewtype
        self.train_file_name = self.base_file_name + '_train'
        self.test_file_name = self.base_file_name + '_test'

        self.raw_train_path = os.path.join('data', 'raw_data',
                                           self.train_file_name + '.xml')
        self.raw_test_path = os.path.join('data', 'raw_data',
                                          self.test_file_name + '.xml')
        self.cleaned_train_path = os.path.join('data', 'cleaned_data',
                                               self.train_file_name + '.csv')
        self.cleaned_test_path = os.path.join('data', 'cleaned_data',
                                              self.test_file_name + '.csv')
        self.datafield_path = os.path.join('data', 'datafields',
                                           self.base_file_name)
        self.ontology_train_result_path = os.path.join('data', 'ontology',
                                                       self.train_file_name)
        self.ontology_test_result_path = os.path.join('data', 'ontology',
                                                      self.test_file_name)

def clean_data(file_names, reload_xml=False):
    datafield_path = file_names.datafield_path

    if reload_xml:
        print('Loading data...')
        load_data(file_names)
        print('Loaded data')
    elif not os.path.isfile(datafield_path):
        print('Data not loaded yet \n Loading data...')
        load_data(file_names)
        print('Loaded data')

    cleaned_train_path = file_names.cleaned_train_path
    cleaned_test_path = file_names.cleaned_test_path
    print('Using cleaned data from ' + cleaned_train_path + ' and ' +
          cleaned_test_path)

def get_batches(device, file_names, batch_sizes, ontology_mask):
    # Load data fields from file
    datafield_path = file_names.datafield_path
    with open(datafield_path, 'rb') as f:
        [text_field, polarity_field, train_test_fields, stats] = dill.load(f)

    # Use only test examples that could not be handled by ontology
    if ontology_mask:
        remaining_train_filter = lambda x: True

        ontology_test_result_path = file_names.ontology_test_result_path
        with open(ontology_test_result_path, 'rb') as f:
            remaining_test = dill.load(f)[0]
        remaining_test_filter = lambda x: int(x.Position) in remaining_test
    else:
        remaining_train_filter = lambda x: True
        remaining_test_filter = lambda x: True

    cleaned_train_path = file_names.cleaned_train_path
    trainds = data.TabularDataset(cleaned_train_path,
                                  format='csv',
                                  csv_reader_params={'delimiter': ',',
                                                     'quotechar': '"'},
                                  fields=train_test_fields,
                                  skip_header=True,
                                  filter_pred=remaining_train_filter)
    cleaned_test_path = file_names.cleaned_test_path
    testds = data.TabularDataset(cleaned_test_path,
                                 format='csv',
                                 csv_reader_params={'delimiter': ',',
                                                    'quotechar': '"'},
                                 fields=train_test_fields,
                                 skip_header=True,
                                 filter_pred=remaining_test_filter)

    traindl = data.BucketIterator(trainds,
                                  batch_sizes[0],
                                  sort_key=lambda x: len(x.SentenceText),
                                  train=True,
                                  sort_within_batch=True,
                                  repeat=False,
                                  device=device)
    testdl = data.BucketIterator(testds,
                                 batch_sizes[1],
                                 sort_key=lambda x: len(x.SentenceText),
                                 train=False,
                                 sort_within_batch=True,
                                 repeat=False,
                                 device=device)
    train_batch_it = BatchGenerator(traindl)
    test_batch_it = BatchGenerator(testdl)
    return train_batch_it, test_batch_it, text_field.vocab, polarity_field, stats

class BatchGenerator:
    def __init__(self, dl):
        self.dl = dl
        self.SentenceText = 'SentenceText'
        self.AspectText = 'AspectText'
        self.AspectPosition = 'AspectPosition'
        self.Polarity = 'Polarity'

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for batch in self.dl:
            sen = getattr(batch, self.SentenceText)
            asp = getattr(batch, self.AspectText)
            asp_position = getattr(batch, self.AspectPosition)
            polarity = getattr(batch, self.Polarity)
            yield sen, asp, asp_position, polarity

def load_data(file_names):
    # Load raw data from .xml files
    raw_train_path = file_names.raw_train_path
    raw_test_path = file_names.raw_test_path
    train_stats = load_xml(raw_train_path)
    test_stats = load_xml(raw_test_path)

    def tokenizer(sentence):
        return [word for word in sentence]

    text_field = data.Field(sequential=True,
                            batch_first=True)
    asp_position_field = data.Field(sequential=False,
                                    batch_first=True)
    polarity_field = data.LabelField(sequential=False,
                                     batch_first=True)
    position_field = data.Field(sequential=False,
                                batch_first=True)

    train_test_fields = [
            ('SentenceText', text_field),
            ('AspectText', text_field),
            ('AspectPosition', asp_position_field),
            ('Polarity', polarity_field),
            ('Position', position_field)]

    cleaned_train_path = file_names.cleaned_train_path
    trainds = data.TabularDataset(cleaned_train_path,
                                  format='csv',
                                  csv_reader_params={'delimiter': ',',
                                                     'quotechar': '"'},
                                  fields=train_test_fields,
                                  skip_header=True)
    cleaned_test_path = file_names.cleaned_test_path
    testds = data.TabularDataset(cleaned_test_path,
                                 format='csv',
                                 csv_reader_params={'delimiter': ',',
                                                    'quotechar': '"'},
                                 fields=train_test_fields,
                                 skip_header=True)

    # Specify the path to the localy saved vectors
    vec = vocab.Vectors('glove.42B.300d.txt', os.path.join('data', 'glove'))
    # Build the vocabulary using the datasets and assign the vectors
    text_field.build_vocab(trainds, testds, max_size=100000, vectors=vec,
                           unk_init = torch.Tensor.normal_)
    asp_position_field.build_vocab(trainds, testds)
    polarity_field.build_vocab(trainds, testds)
    position_field.build_vocab(trainds, testds)

    # Write the datafields to dill files
    datafield_path = file_names.datafield_path
    with open(datafield_path, 'wb') as f:
        dill.dump([text_field, polarity_field, train_test_fields,
                   [train_stats, test_stats]], f)


def load_xml(path):
    with open(path, 'r', encoding='utf-8') as xml_file:
        tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    pos = 0

    n_review = 0
    n_sen = 0
    n_opinion = 0
    n_target = 0
    n_pos = 0
    n_neut = 0
    n_neg = 0

    nlp = spacy.load('en',disable=['parser', 'tagger', 'ner'])
    def text_clean(text):
        # Remove non-alphanumeric characters
        text = re.sub(r'[^A-Za-z0-9]+', ' ', text)
        return nlp(text.strip()).text.lower()

    def aspect_position(sen, asp):
        "Find the position of the aspects in the sentence."
        split_sen = sen.split()
        split_asp = asp.split()
        sen_len = len(split_sen)
        asp_len = len(asp.split())
        for position, word in enumerate(split_sen):
            end = position + asp_len
            if end <= sen_len:
                sub_string = split_sen[position:end]
                if sub_string == split_asp:
                    return [position, end]

    for review in root.iter('Review'):
        n_review += 1
        for sentence in review.iter('sentence'):
            n_sen += 1
            sen = sentence.find('text').text
            sen_clean = text_clean(sen)
            for opinion in sentence.iter('Opinion'):
                n_opinion += 1
                if opinion.get('target') != "NULL":
                    n_target += 1
                    asp_from = int(opinion.get('from'))
                    asp_to = int(opinion.get('to'))
                    asp = text_clean(sen[asp_from:asp_to])
                    asp_position = aspect_position(sen_clean, asp)
                    if opinion.get('polarity') == 'positive':
                        n_pos += 1
                        polarity = 'positive'
                    elif opinion.get('polarity') == 'neutral':
                        n_neut += 1
                        polarity = 'neutral'
                    else:
                        n_neg += 1
                        polarity = 'negative'

                    data.append([sen_clean, asp, asp_position, polarity, pos])
                    pos += 1

    csv_file_name = os.path.join('data', 'cleaned_data',
                        os.path.splitext(os.path.basename(path))[0] + '.csv')
    with open(csv_file_name, mode = 'w', encoding='utf-8') as data_file:
        data_writer = csv.writer(data_file)
        data_writer.writerow(['Sentence text', 'Aspect text',
                              'Aspect position', 'Polarity', 'Position'])
        data_writer.writerows(data)
    return [n_review, n_sen, n_opinion, n_target, n_pos, n_neut, n_neg]