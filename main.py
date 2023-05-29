import sys

import numpy as np

from data import load_data, add_new_columns, data_analysis
from clustering import kmeans


def main(argv):
    path = argv[1]
    df = load_data(path)

    print("Part A: ")
    df = add_new_columns(df)
    data_analysis(df)
#TODO: uncomment

    print("Part B: ")
    df = load_data(path)
    centroids, labels = kmeans(df, 2)
    print(np.array_str(centroids, precision=1, suppress_small=True))




if __name__ == '__main__':
    main(sys.argv)
