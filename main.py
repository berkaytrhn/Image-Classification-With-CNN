
import matplotlib.pyplot as plt

def plotting(train, test, text, batch, lr, status):
    plt.plot(range(1, len(train) + 1), train, 'r', label="Train {}".format(text))
    plt.plot(range(1, len(test) + 1), test, 'b', label="Test {}".format(text))

    plt.title(f"{batch} Batch and {lr} Learning Rate {status}")
    plt.xlabel('Epoch')
    plt.ylabel(text)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    pass
