from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import pandas as pd
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="DP Data Synthesizer - PriveBayes")
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default="./bank-data/bank-additional-full.csv",
        help="Path to input data",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default="./bank-data/synth/random_mode/",
        help="Path to input data",
    )
    args = parser.parse_args()
    df = pd.read_csv(args.input_file, sep=';')

    # input to DataSynthetizer must be comma separated. Create a temp file.
    df.to_csv('comma_data.csv', sep=',')
    input_data = 'comma_data.csv'

    description_file = args.output_dir + '/description.json'
    synthetic_data = args.output_dir + '/sythetic_data.csv'

    # An attribute is categorical if its domain size is less than this threshold.
    # Here modify the threshold to adapt to the domain size of "education" (which is 14 in input dataset).
    threshold_value = 20

    # Number of tuples generated in synthetic dataset.
    num_tuples_to_generate = len(df)


    # Data describer
    describer = DataDescriber(category_threshold=threshold_value)
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

if __name__ == "__main__":
    main()