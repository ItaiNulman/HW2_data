import numpy as np

from data import load_data, add_new_columns, data_analysis
from clustering import kmeans, transform_data, visualize_results


def main():
    path = 'london.csv'
    pdf1 = 'pic1.png'
    pdf2 = 'pic2.png'
    pdf3 = 'pic3.png'
    df = load_data(path)

    print("Part A: ")
    df = add_new_columns(df)
    data_analysis(df)


    print("Part B: ")
    df = load_data(path)
    td = transform_data(df, ['cnt', 'hum'])

    labels1, centroids1 = kmeans(td, 2)
    print("k = 2")
    print(np.array_str(centroids1, precision=3, suppress_small=True))
    visualize_results(td, labels1, centroids1, pdf1)
    print()
    labels2, centroids2 = kmeans(td, 3)
    print("k = 3")
    print(np.array_str(centroids2, precision=3, suppress_small=True))
    visualize_results(td, labels2, centroids2, pdf2)
    print()
    labels3, centroids3 = kmeans(td, 5)
    print("k = 5")
    print(np.array_str(centroids3, precision=3, suppress_small=True), end='')
    visualize_results(td, labels3, centroids3, pdf3)












if __name__ == '__main__':
    main()
