# NOTE: run using Python2, pandas 0.19.0, and tables
import pandas as pd
import glob
import os


# Input the folders where you downloaded the data from
# http://dx.doi.org/10.17632/rssf5nxxby.3 here. We have here
# assumed the name of the folder "LGG_ARTICLE_DATA"
datafolders = [os.path.join('LGG_ARTICLE_DATA', 'FEATURES', 'TRAINING'),
               os.path.join('LGG_ARTICLE_DATA', 'FEATURES', 'TESTING')]

for datafolder in datafolders:
    featurefiles = glob.glob(os.path.join(datafolder, 'features*.hdf5'))
    featurefiles.sort()

    for featurefile in featurefiles:
        print('Processing ' + featurefile)
        # Read features
        data = pd.read_hdf(featurefile)
        data = data.image_features

        # Gather labels and values
        feature_labels = list()
        feature_values = list()
        for key, value in data['MR_T1_texture_features'].iteritems():
            key = 'tf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value[0])

        for key, value in data['MR_T1_histogram_features'].iteritems():
            key = 'hf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        for key, value in data['MR_T2_texture_features'].iteritems():
            key = 'tf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value[0])

        for key, value in data['MR_T2_histogram_features'].iteritems():
            key = 'hf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        for key, value in data['patient_features'].iteritems():
            key = 'semf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        for key, value in data['Location'].iteritems():
            key = 'of_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        for key, value in data['Shape_2D'].iteritems():
            key = 'sf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        for key, value in data['Shape_3D'].iteritems():
            key = 'sf_original_PREDICT_' + key
            feature_labels.append(key)
            feature_values.append(value)

        # Convert to csv
        csvfile = os.path.join(os.path.dirname(featurefile) + '_WORC3', os.path.splitext(os.path.basename(featurefile))[0] + '.csv')
        data = pd.DataFrame({'Labels': feature_labels,
                             'Values': feature_values}, index=range(len(feature_labels)))
        data.to_csv(csvfile, index=False)
