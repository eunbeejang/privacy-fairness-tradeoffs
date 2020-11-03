from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="DP Data Synthesizer - PriveBayes")
    """
    parser.add_argument( # remove/change
        "-i",
        "--input-file",
        type=str,
        default="./bank-data/bank-additional-full.csv",
        help="Path to input data",
    )
    parser.add_argument( # remove/change
        "-o",
        "--output-dir",
        type=str,
        default="./bank-data/synth/",
        help="Path to input data",
    )
    """
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="independent",
        help="Synthesizer Mode: 'independent', 'correlated', 'random'",
    )
    parser.add_argument(
        "-d",
        "--data",
        type=str,
        default="bank",
        help="Dataset ('bank' for bank dataset, 'adult' for adult dataset)",
    )
    parser.add_argument(
        "-e",
        "--epsilon",
        type=float,
        default=0.1,
        help="Noise parameter (default = 0.1)",
    )
    args = parser.parse_args()
    if args.data == 'bank':
        input_file = './bank-data/bank-additional-full.csv'
        df = pd.read_csv(input_file, sep=';')
        # specify categorical attributes
        categorical_attributes = {'age': True,
                                  'job': True,
                                  'marital': True,
                                  'education': True,
                                  'default': True,
                                  'housing': True,
                                  'loan': True,
                                  'contact': True,
                                  'month': True,
                                  'day_of_week': True,
                                  'poutcome': True,
                                  'y': True
                                  }
        output_dir = './bank-data/synth'
        sep = ';'
    elif args.data == 'adult':
        input_file = './adult-data/adult.data'
        cols = ['age', 'workclass', 'fnlwgt', 'education',
                'education-num', 'marital-status', 'occupation',
                'relationship', 'race', 'sex', 'capital-gain',
                'capital-loss', 'hours-per-week', 'native-country', 'y']

        df = pd.read_csv(input_file, sep=',', names=cols)
        categorical_columns = ['workclass', 'education', 'marital-status',
                               'occupation', 'relationship', 'race',
                               'sex', 'native-country']
        for category in categorical_columns:
            df[category] = df[category].astype('object')

        categorical_attributes = {'workclass': True,
                                  'education': True,
                                  'marital-status': True,
                                  'occupation': True,
                                  'relationship': True,
                                  'race': True,
                                  'sex': True,
                                  'native-country': True,
                                  'y': True
                                  }
        output_dir = './adult-data/synth'
        sep = ','

    #df = df.dropna()

    # input to DataSynthetizer must be comma separated. Create a temp file.
    df.to_csv('comma_data.csv', sep=',')
    input_data = 'comma_data.csv'

    description_file = output_dir + '/description.json'
    synthetic_data = ''
    save_path = ''
    # An attribute is categorical if its domain size is less than this threshold.
    # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
    threshold_value = 20

    # Number of tuples generated in synthetic dataset.
    num_tuples_to_generate = len(df)


    # specify which attributes are candidate keys of input dataset.
    candidate_keys = {'y': 'yes'}


    # Increase epsilon value to reduce the injected noises. Set epsilon=0 to turn off differential privacy.
    epsilon = args.epsilon

    # The maximum number of parents in Bayesian network, i.e., the maximum number of incoming edges.
    degree_of_bayesian_network = 2

    # Number of tuples generated in synthetic dataset.
    num_tuples_to_generate = len(df)

    # Data describer
    describer = DataDescriber(category_threshold=threshold_value)
    if args.mode == 'independent':
        synthetic_data = output_dir + '/syth_data_independent.csv'
        save_path = output_dir + '/syth_data_independent_ymod.csv'

        describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_data,
                                                                 attribute_to_is_categorical=categorical_attributes,
                                                                 attribute_to_is_candidate_key=candidate_keys)

        describer.save_dataset_description_to_file(description_file)

    elif args.mode == 'correlated':
        synthetic_data = output_dir + '/syth_data_correlated.csv'
        save_path = output_dir + '/syth_data_correlated_ymod.csv'

        describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                                epsilon=epsilon,
                                                                k=degree_of_bayesian_network,
                                                                attribute_to_is_categorical=categorical_attributes,
                                                                attribute_to_is_candidate_key=candidate_keys)

        describer.save_dataset_description_to_file(description_file)

        print(display_bayesian_network(describer.bayesian_network))

    else:
        synthetic_data = output_dir + '/syth_data_random.csv'
        save_path = output_dir + '/syth_data_random_ymod.csv'

        describer.describe_dataset_in_random_mode(input_data)

        describer.save_dataset_description_to_file(description_file)

    # Generate synthetic dataset
    generator = DataGenerator()
    generator.generate_dataset_in_random_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

    """
    # Compare the stats of original and synthetic data
    # Read both datasets using Pandas.
    input_df = pd.read_csv(input_data, skipinitialspace=True)
    synthetic_df = pd.read_csv(synthetic_data)
    # Read attribute description from the dataset description file.
    attribute_description = read_json_file(description_file)['attribute_description']

    inspector = ModelInspector(input_df, synthetic_df, attribute_description)
    for attribute in synthetic_df.columns:
        inspector.compare_histograms(attribute)
    """

    # Delete temporary file (comma separated df)
    if os.path.exists(input_data):
        os.remove(input_data)


    synth_df = pd.read_csv(synthetic_data, sep=',')
    synth_df['y'] = df['y']
    save_df = synth_df.loc[:, 'age':'y']

    save_df.to_csv(save_path, sep=sep, index=False, header=None)

if __name__ == "__main__":
    main()