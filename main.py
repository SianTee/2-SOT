import os

import dill
import numpy as np
import torch

from data_reader import FileNames, clean_data, get_batches
from ontology_reasoner import OntReasoner
from transformer import run_transformer

# Set device for machine learning
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Choose model settings
YEAR = 16               # 15 or 16 for 2015 or 2016 data
RELOAD_DATA = True      # True: load, clean and create vocabulary from raw data
                        # False: load prepared data (or reload if unavailable)
RERUN_ONTOLOGY = True   # True: run the ontology and get remaining  examples
                        # False: use ontology results from pre-run file
RANDOM_SEED = 1234
ONTOLOGY_MASK = True

# Choose hyperparameters
LEARNING_RATE = 1e-2    # Learning rate for Adam optimizer
DROPOUT = 0.5
EPOCHS = 10             # Number of times to go through complete dataset once
                        # #iterations = #epochs * #batches in 1 epoch
TRAIN_BATCH_SIZE = 20
TEST_BATCH_SIZE = 200

N_SEN = 2               # #encoder layers for sentence in Transformer
N_ASP = 2               # #encoder layers for aspect in Transformer

# #dimensions of model (300) should be divisible by #heads
H_SEN = 3               # #heads for sentence in Transformer
H_ASP = 3               # #heads for aspect in Transformer

D_FF1 = 500             # #dimensions in FFN in Transformer
D_FF2 = 50              # #dimensions in FFN on aspect-specific sentence


batch_sizes = (TRAIN_BATCH_SIZE, TEST_BATCH_SIZE)

file_names = FileNames(YEAR)

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Clean data and store in .csv files
clean_data(file_names, RELOAD_DATA)

if RERUN_ONTOLOGY or not os.path.isfile(file_names.ontology_train_result_path):
    print('Running ontology on train data...')
    # Run ontology on train set
    ontology = OntReasoner()
    train_accuracy_ont, train_correct_ont, train_size, train_remain_ont = \
        ontology.run(file_names.cleaned_train_path)

    # Run ontology on test set
    print('Running ontology on test data...')
    ontology = OntReasoner()
    test_accuracy_ont, test_correct_ont, test_size, test_remain_ont = \
        ontology.run(file_names.cleaned_test_path)
else:
    print('Loading ontology results...')
    # Load ontology results from file
    ontology_train_result_path = file_names.ontology_train_result_path
    with open(ontology_train_result_path, 'rb') as f:
        [_, train_accuracy_ont, train_correct_ont, train_size, train_remain_ont] = dill.load(f)
    ontology_test_result_path = file_names.ontology_test_result_path
    with open(ontology_test_result_path, 'rb') as f:
        [_, test_accuracy_ont, test_correct_ont, test_size, test_remain_ont] = dill.load(f)

train_batch_it, test_batch_it, vocab, polarity_field, stats = get_batches(device, file_names, batch_sizes, ONTOLOGY_MASK)
[[train_rev, train_sen, train_opinion, train_tar, train_pos, train_neut, train_neg],
 [test_rev, test_sen, test_opinion, test_tar, test_pos, test_neut, test_neg]] = stats

n_train_ml = len(next(iter(train_batch_it.dl)).dataset)
n_test_ml = len(next(iter(test_batch_it.dl)).dataset)
print(f'ML training set size: {n_train_ml:d}')
print(f'ML test set size: {n_test_ml:d}')
model, train_loss, test_loss, train_accuracy_ml, test_accuracy_ml, train_correct_ml, test_correct_ml = run_transformer(device, LEARNING_RATE, EPOCHS, DROPOUT, N_SEN, N_ASP, H_SEN, H_ASP, D_FF1, D_FF2,
     train_batch_it, test_batch_it, vocab)

print('Model finished\n')
print('Data statistics:')
print(f'\tReview\tSent.\tOpin.\tTarget\tPos.\tNeut.\tNeg.')
print(f'Train\t{train_rev:d}\t{train_sen:d}\t{train_opinion:d}\t{train_tar:d}\t{train_pos:d}\t{train_neut:d}\t{train_neg:d}')
print(f'Test\t{test_rev:d}\t{test_sen:d}\t{test_opinion:d}\t{test_tar:d}\t{test_pos:d}\t{test_neut:d}\t{test_neg:d}\n')

print('Ontology statistics:')
print(f'Ontology training set size: {train_size:d} opinions')
print(f'Ontology test set size: {test_size:d} opinions')
print(f'\tOntology train solved: {train_size-train_remain_ont:4d} of {train_size:4d}'
      f' | Ontology train acc.: {train_accuracy_ont*100:.2f}%')
print(f'\t Ontology test solved: {test_size-test_remain_ont:4d} of {test_size:4d}'
      f' |  Ontology test acc.: {test_accuracy_ont*100:.2f}%\n')

print('Transformer statistics:')
print(f'ML training set size: {n_train_ml:d} opinions')
print(f'ML test set size: {n_test_ml:d} opinions')
print(f'\tML train loss: {train_loss:.3f}'
      f' | ML train acc.: {train_accuracy_ml*100:.2f}%')
print(f'\t ML test loss: {test_loss:.3f}'
      f' |  ML test acc.: {test_accuracy_ml*100:.2f}%\n')

train_correct_total = train_correct_ont + train_correct_ml
train_accuracy_total = train_accuracy_ml
test_correct_total = test_correct_ont + test_correct_ml
test_accuracy_total = test_correct_total / test_size if ONTOLOGY_MASK else test_accuracy_ml
print('Overall model statistics:')
print(f'\t Overall train acc.: {train_accuracy_total*100:.2f}%')
print(f'\t  Overall test Acc.: {test_accuracy_total*100:.2f}%\n')