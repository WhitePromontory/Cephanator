import torch
from torch import nn
from torchvision import models

class CephNet(nn.Module):
    """
    CephNet Model, comprised of ResNet50 and a modified FC layer

    Input: Batch of RGB 512 x 512 cephalograms, (Batch, 3, 512, 512)

    Output: A batch of 29 landmarks, defined by co-ordinates (Batch, 29, 2)



    """
    def __init__(self):
        super().__init__()
        self.ResNet = models.resnet50(weights="DEFAULT")
        # features from original fc later
        num_features = self.ResNet.fc.in_features
        # Re-assign ResNet FC layer
        self.ResNet.fc = nn.Linear (num_features, 58)
        # Number of flattened features (b, 58)

    def forward(self, x):
        # (b,58)
        flat_output = self.ResNet(x)
        # (1, 29, 2)
        output = flat_output.view(-1,29,2)

        return output


def train(model, dataloader, loss_fn, optimizer, device, i):
    """

    :param model: CephNet model
    :param dataloader: Training Dataloader
    :param loss_fn: Loss Function
    :param optimizer: Optimization Function
    :param device: GPU or CPU
    :param i: Number of Epochs
    :return: Average loss, Euclidean distance per image
    """

    model.train()
    train_loss = 0
    total_euclid = 0

    for batch_idx, (input, targets) in enumerate(dataloader):
        # Every data instance is an input + label pair
        input = input.to(device)
        target = targets.to(device)

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        output = model(input)

        # Compute the loss and its gradients
        loss = loss_fn(output, target)
        batch_mre = calculate_mre(output, target)

        # backprop
        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # average batch loss * batch number
        train_loss += loss.item() * input.size(0)
        total_euclid += batch_mre.item() * input.size(0)


    # average loss per image in epoch
    epoch_loss = train_loss / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Training Loss: {epoch_loss:.6f}")
    avg_euclid = total_euclid / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Euclidean Distance: {avg_euclid:.4f}")

    return epoch_loss, avg_euclid


def validate (model, dataloader, loss_fn,device, i):
    """

    :param model: CephNet model
    :param dataloader: Validation Dataloader
    :param loss_fn: Loss Function
    :param device: GPU or CPU
    :param i: Number of Epochs
    :return: Average loss, Euclidean distance per image
    """
    model.eval()
    val_loss = 0
    total_euclid = 0
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(dataloader):
            # Every data instance is an input + label pair
            input = input.to(device)
            target = target.to(device)

            # Make predictions for this batch
            output = model(input)

            # Compute the loss and its gradients
            loss = loss_fn(output, target)
            batch_mre = calculate_mre(output, target)


            # Average val of batch * batch num
            val_loss += loss.item() * input.size(0)
            total_euclid += batch_mre * input.size(0)

    # average loss per image
    epoch_loss = val_loss / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Val Loss: {epoch_loss:.6f}")
    avg_euclid = total_euclid / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Euclidean Distance: {avg_euclid:.4f}")

    return val_loss, avg_euclid


def test (model, dataloader, loss_fn,device, i):
    """

    :param model: CephNet Model
    :param dataloader: Test Dataloader
    :param loss_fn: Loss Function
    :param device: GPU or CPU
    :param i: Number of Epochs
    :return: Average loss, Euclidean distance per image
    """

    model.eval()
    test_loss = 0
    total_euclid = 0

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(dataloader):
            # Every data instance is an input + label pair
            input = input.to(device)
            target = target.to(device)


            # Make predictions for this batch
            output = model(input)

            # Compute the loss and its gradients
            loss = loss_fn(output, target)

            # Average val of batch * batch num
            test_loss += loss.item() * input.size(0)

            batch_mre = calculate_mre(output, target)
            total_euclid += batch_mre * input.size(0)

    # average loss per image
    epoch_loss = test_loss / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Val Loss: {epoch_loss:.6f}")
    avg_euclid = total_euclid / len(dataloader.dataset)
    print(f"Epoch: {i} | Average Euclidean Distance: {avg_euclid:.4f}")

    return test_loss,avg_euclid


def calculate_mre(predictions, targets):
    """
    Calculates Euclidean Distance of predicted and target landmark coordinates

    Shapes: (Batch, 29, 2)
    """
    # 1. Calculate (delta_x, delta_y)
    # 2. (delta_x)^2 + (delta_y)^2
    # 3. Aggregate Sum x^2 and y^2 along the last dimension (dim=-1), sigma 1 to b ((x^2+y^2))
    # 4. Square root to get the Euclidean distance of each batch
    distances = torch.sqrt(torch.sum((predictions - targets) ** 2, dim=-1))

    # Return the average distance across all landmarks in the batch
    return distances.mean()
