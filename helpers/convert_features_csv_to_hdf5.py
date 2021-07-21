# NOTE: run in Python 3 with pandas package (any version) installed
import pandas as pd
import glob
import os

# Input the folders where the processed features from
# convert_features_hdf5_to_csv script are stored here
datafolders = [os.path.join('LGG_ARTICLE_DATA', 'FEATURES', 'TRAINING_WORC3'),
               os.path.join('LGG_ARTICLE_DATA', 'FEATURES', 'TESTING_WORC3')]


panda_labels = ['feature_values', 'feature_labels']
for datafolder in datafolders:
    featurefiles = glob.glob(os.path.join(datafolder, 'features*.csv'))
    featurefiles.sort()

    for featurefile in featurefiles:
        print('Processing ' + featurefile)

        # Read features
        data = pd.read_csv(featurefile)
        feature_labels = data['Labels'].values
        feature_values = data['Values'].values

        # Convert
        panda_data = pd.Series([feature_values.tolist(), feature_labels.tolist()],
                               index=panda_labels,
                               name='Image features'
                               )

        # Write output
        outputfile = os.path.splitext(featurefile)[0] + '.hdf5'
        panda_data.to_hdf(outputfile, 'image_features')
