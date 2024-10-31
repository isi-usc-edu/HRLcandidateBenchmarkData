import pandas as pd
import matplotlib.pyplot as plt



if __name__ == "__main__":

    # Load the CSV file into a DataFrame
    file_path = 'emperical_metrics.csv'  # Replace 'your_file.csv' with the path to your CSV file
    df = pd.read_csv(file_path)

    print(df.columns)

    # Plot each column in the DataFrame against 'norb'
    plt.figure()
    for i ,column in enumerate(df.columns):

        if column != 'n_orbs' and i > 0:

            plt.scatter(df['n_orbs'], df[column], marker='o', linestyle='-', label=column)


    plt.grid(True)
    plt.show()