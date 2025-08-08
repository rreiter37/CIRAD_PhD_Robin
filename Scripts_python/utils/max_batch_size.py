import torch

def find_max_batch_size(model, input_shape, device="cuda", max_batch=2048, min_batch=1):
    """
    Find the maximum batch size that fits into GPU memory without raising an OOM error.

    Args:
        model (torch.nn.Module): The model to test.
        input_shape (tuple): Shape of a single input (excluding batch dimension), e.g., (3, 224, 224).
        device (str): Device to run on ("cuda" or "cpu").
        max_batch (int): Maximum batch size to try.
        min_batch (int): Minimum batch size to try.

    Returns:
        int: The largest batch size that does not raise a CUDA OOM error.
    """
    model = model.to(device)
    batch_size = max_batch

    while batch_size >= min_batch:
        try:
            # Create a dummy input batch
            dummy_input = torch.randn((batch_size, *input_shape), device=device)
            with torch.no_grad():
                _ = model(dummy_input)
            torch.cuda.empty_cache()
            return batch_size
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                batch_size = batch_size // 2
            else:
                raise e

    raise RuntimeError(f"Unable to find a working batch size between {min_batch} and {max_batch}")
