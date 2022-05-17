import sys

from sklearn.model_selection import train_test_split

sys.path.append('../')
import pandas as pd
from tqdm import tqdm
from utils.basic_utils import dump_pickle, load_pickle
from utils.display_utils import info
import gc
import numpy as np

# 30_classes = 33 classes + Silence + Speech - 'Glass' - 'Meow' - 'Pig' - 'Wild animals' - 'Silence'
classes30 = ['Speech', 'Baby cry, infant cry', 'Cough', 'Sneeze', 'Walk, footsteps', 'Clapping', 'Cheering', 'Dog', 'Cat', 'Cowbell',  'Chicken, rooster', 'Frog', 'Music', 'Wind', 'Rain', 'Raindrop', 'Rain on surface', 'Ocean', 'Waves, surf', 'Vehicle', 'Helicopter', 'Engine', 'Chainsaw', 'Tap', 'Water tap, faucet', 'Keys jangling', 'Clock', 'Firecracker',  'Scratch', 'Squeal']
as_label_idx = pd.read_csv('/home/alhasan/workdir/SED/data/AudioSet/class_labels_indices.csv')
as_label_idx = as_label_idx.loc[as_label_idx['display_name'].isin(classes30)]
classes30_idx = as_label_idx['index'].to_list()


labels_dict = load_pickle('/home/alhasan/workdir/SED/data/AudioSet/PANN_labels_all.pkl')['labels']
embeddings_dict = load_pickle('/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_20_512.pkl')['embeddings']
files = [f for f in embeddings_dict.keys() if f in labels_dict.keys()]


embeddings =[]
as_labels = []

for f in tqdm(files):
    n_segments = min(len(embeddings_dict[f]), len(labels_dict[f]))
    embeddings.append(embeddings_dict[f][:n_segments])
    as_labels.append(labels_dict[f][:n_segments])

# concatenate
as_labels = np.concatenate(as_labels, axis=0)
embeddings = np.concatenate(embeddings, axis=0)

info(f"len files labels: {len(as_labels)}")
info(f"embeddings: {len(embeddings)}")
# save data
panns_dataset = {
                'labels': as_labels,
                'embeddings': embeddings
                 }
dump_pickle(panns_dataset, '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_500_512_panns.pkl')


classes_30_samples = []
info ("Collecting G30...")
for i in tqdm(range(len(as_labels))):
    if as_labels[i] in classes30_idx:
        classes_30_samples.append(i)
info(f"samples to use: {len(classes_30_samples)}")
labels30 = [as_labels[i] for i in classes_30_samples]
embeddings30 = [embeddings[i] for i in classes_30_samples]

assert(len(set(labels30)) == 30), f"Expected 30 classes got {len(set(labels30))}"
info(f"Numbre of labels in G30: {len(set(labels30))}")
info(f"Numbre of samples in G30: {len(set(labels30))}")

panns_dataset30 = {
                'classes_30': classes30,
                'classes30_audioset_index': classes30_idx,
                'labels': labels30,
                'embeddings': embeddings30
                 }
info("Saving G30 to disk")
ds30_path = '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_500_512_panns(classes_30).pkl'
dump_pickle(panns_dataset30, ds30_path)

del embeddings, as_labels, panns_dataset30
gc.collect()

# split data for train and test
X_train, X_test, y_train, y_test = train_test_split(embeddings30, labels30, test_size=0.15, random_state=49)
panns_dataset30_train = {
                'classes_30': classes30,
                'classes30_audioset_index': classes30_idx,
                'labels': y_train,
                'embeddings': X_train
                 }
panns_dataset30_test = {
                'classes_30': classes30,
                'classes30_audioset_index': classes30_idx,
                'labels': y_test,
                'embeddings': X_test
                 }                 
info("Saving G30 train/test to disk")
ds30_train_path = '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_500_512_panns(classes_30)_train.pkl'
ds30_test_path = '/home/alhasan/workdir/SED/data/AudioSet/openl3_embeddings_500_512_panns(classes_30)_test.pkl'

dump_pickle(panns_dataset30_train, ds30_train_path)
dump_pickle(panns_dataset30_test, ds30_test_path)


# info("30 classes count ", len([x for x in as_labels if x in classes30_idx]))                                                  # 35 classes count   3859583
# info("all samples count", len(as_labels))                                                                                     # all samples count  4922630
# info("differece count  ", len(as_labels) - len([x for x in as_labels if x in classes30_idx]))                                 # differece count    1063052
# info("diff %           ", (len(as_labels) - len([x for x in as_labels if x in classes30_idx]))/len(as_labels)*100)            # diff %             21.605%
