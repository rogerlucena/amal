from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, Linear, ReLU, Sigmoid
from datamaestro import prepare_dataset


class MyDataset(Dataset):
    def __init__(self, images, labels):
        assert(len(images) == len(labels))
        self.length = len(images)
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return self.length


class Autoencoder(Module):
    def __init__(self):
        self.fc1 = Linear(784, 400)
        # self.fc2 is the transpose of fc1 params

    def encode(self, x):
        return ReLU(self.fc1(x))

    def decode(self, z):
        return Sigmoid(self.fc2.())


if __name__ == '__main__':
    BATCH_SIZE = 10

    ds = prepare_dataset("com.lecun.mnist")

    train_images, train_labels = ds.files["train/images"].data(
    ), ds.files["train/labels"].data()

    test_images, test_labels = ds.files["test/images"].data(
    ), ds.files["test/labels"].data()

    train_data = DataLoader(MyDataset(train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)
    
    for x, y in train_data:
        print(y)
        

