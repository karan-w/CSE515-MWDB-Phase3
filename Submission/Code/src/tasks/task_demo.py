import csv
import time

import numpy as np
import pandas as pd


def pageRank(graph, input_images, beta=0.85, epsilon=0.000001):
    nodes = len(graph)

    # Dictionaries mapping index number of adjacency matrix to the image IDs and vice-a-versa
    index_img_dict = dict(csv.reader(open('IndexToImage.csv', 'r')))
    img_index_dict = dict([[v, k] for k, v in index_img_dict.items()])

    M = graph / np.sum(graph, axis=0)

    # Initializing Teleportation matrix and Page Rank Scores with Zeros for all images
    teleportation_matrix = np.zeros(nodes)
    pageRankScores = np.zeros(nodes)


    # Updating Teleportation and Page Rank Score Matrices with 1/num_of_input images for the input images.
    for image_id in input_images:
        teleportation_matrix[int(img_index_dict[image_id])] = 1 / len(input_images)
        pageRankScores[int(img_index_dict[image_id])] = 1 / len(input_images)

    # Calculating Page Rank Scores
    while True:
        oldPageRankScores = pageRankScores
        pageRankScores = (beta * np.dot(M, pageRankScores)) + ((1 - beta) * teleportation_matrix)
        if np.linalg.norm(pageRankScores - oldPageRankScores) < epsilon:
            break



    # Normalizing & Returning Page Rank Scores
    return pageRankScores / sum(pageRankScores)


def main():
    print('Loading Image-Image Similarity Matrix...')

    # Loading the ImageSimilarities.csv file
    graph = pd.read_csv("ImageSimilarities.csv", index_col=0).values

    # Dictionary mapping index number of adjacency matrix to the image IDs
    index_img_dict = dict(csv.reader(open('IndexToImage.csv', 'r')))

    labels = set()

    # Reads user input file
    input_image_label_pair = []
    with open('input.txt', 'r') as file:
        for line in file:
            image_id = line.strip().split(' ')[0]
            label = line.strip().split(' ')[-1]
            labels.add(label)
            input_image_label_pair.append([image_id, label])

    # Calculating Personalized Page Rank for each label once.
    output = []
    label_dict = dict()
    count = 0
    for label in labels:
        label_dict[str(count)] = label
        count += 1
        print('Calculating Personalised Page Rank for label : ' + label)
        input_images = [item[0] for item in input_image_label_pair if item[1] == label]
        output.append([label, pageRank(graph, input_images, beta=0.85).tolist()])

    # Assigning label to each image based on the highest Page Rank score value
    image_labels = []
    for i in range(0, len(output[0][1])):
        compare = []
        for item in output:
            compare.append([item[0], i, item[1][i]])
        image_labels.append(sorted(compare, reverse=True, key=lambda x: x[2])[0])

    # Grouping images for each labels
    final_output = dict()
    for elem in image_labels:
        if elem[0] not in final_output:
            final_output[elem[0]] = []
        final_output[elem[0]].append(index_img_dict[str(elem[1])])
    print(final_output)

    results = []
    for k, v in label_dict.items():
        results.append(final_output[v])
    # Printing Output
    print(results)


if __name__ == '__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('Time : ', end_time - start_time)