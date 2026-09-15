from train import main, plot_losses
from GPTModel import GPTModel
import torch
import matplotlib.pyplot as plt

if __name__=='__main__':
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # vocabulary size
        "context_length": 256,  # shortened context length
        "emb_dim": 768,         # embedding dimension
        "n_heads": 12,          # number of attention heads
        "n_layers": 12,         # number of layers
        "drop_rate": 0.1,       # dropout rate
        "qkv_bias": False       # qkv-bias
    }

    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 3,
        "batch_size": 2,
        "weight_decay": 0.1
    }

    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    # plot results
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # Save and load model
    torch.save(model.state_dict(), "model.pth")
    model = GPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load("model.pth", weights_only=True))
