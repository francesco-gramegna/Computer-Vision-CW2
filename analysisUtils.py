import matplotlib.pyplot as plt
import numpy as np

def plotConfusion(classifier, testX, testY, classMappings):
    allClasses = list(classMappings.items())

    confusion = {true[1]: {pred[1]: 0 for pred in allClasses} for true in allClasses}

    for img, _class in zip(testX,testY):
        pred = classifier.predict([img])
        pred = pred[0]
        confusion[_class][pred]+=1


     # Convert to matrix
    matrix = np.array([[confusion[true[1]][pred[1]] for pred in allClasses] for true in allClasses])

    # Plot matrix
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(matrix)

    # Add labels
    ax.set_xticks(np.arange(len(allClasses)))
    ax.set_yticks(np.arange(len(allClasses)))
    ax.set_xticklabels([c[0] for c in allClasses], rotation=45, ha='right')
    ax.set_yticklabels([c[0] for c in allClasses])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # Add text annotations
    #for i in range(len(allClasses)):
    #    for j in range(len(allClasses)):
    #        ax.text(j, i, matrix[i, j], ha='center', va='center',
    #                color='black' if matrix[i, j] < matrix.max() / 2 else 'white')

    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()




