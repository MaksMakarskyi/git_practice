

def predict_and_print(model, input):
    """

    Args:
        model (nn.Module): _description_
        input (torch.Tensor): _description_
    """
    model.eval()
    print(model(input))