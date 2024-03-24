from collections import OrderedDict
import logging
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_MMCV = "./release_model/E2FGVI-HQ-CVPR22.pth"
CHECKPOINT = "./release_model/E2FGVI-HQ-Nuke.pth"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
MODEL_FILE = "./nuke/Cattery/E2FGVI/E2FGVI.pt"

torch.set_printoptions(precision=10, sci_mode=False)


def load_e2fgvi():
    from model import e2fgvi_hq

    state_dict = torch.load(CHECKPOINT, map_location=DEVICE)

    e2fgvi_model = e2fgvi_hq.InpaintGenerator()
    e2fgvi_model.to(DEVICE)
    e2fgvi_model.load_state_dict(state_dict)
    e2fgvi_model.eval()

    LOGGER.info("E2FGVI HQ Model Loaded.")
    LOGGER.info(e2fgvi_model)
    return e2fgvi_model


class E2fgviNuke(torch.nn.Module):
    def __init__(self, n_frames: int = 6, neighbor: int = 6):
        super().__init__()
        self.n_frames = n_frames
        self.neighbor = neighbor
        self.e2fgvi = load_e2fgvi()

    def forward(self, x: torch.Tensor):
        n_frames = self.n_frames
        neighbor = self.neighbor

        b, c, h, w = x.shape
        image_width: int = w // n_frames
        device = torch.device("cuda") if x.is_cuda else torch.device("cpu")

        # Force input to float32
        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        if x.device != device:
            x = x.to(device)

        unwrapped_input = x.reshape(b, c, h, n_frames, image_width).permute(
            0, 3, 1, 2, 4
        )
        pred_imgs, _ = self.e2fgvi(unwrapped_input, neighbor)

        # Return as an wide image
        return (
            pred_imgs.unsqueeze(0)
            .permute(0, 2, 3, 1, 4)
            .reshape(1, c, h, n_frames * image_width)
        )


def convert_mmcv_to_torch(state_dict: str):
    """Convert mmcv state dict to torch format
    Args:
        state_dict: State dict from mmcv
    Returns:
        state_dict: Converted state dict
    """
    state_dict = torch.load(CHECKPOINT_MMCV)
    new_state_dict = OrderedDict()
    mapping = {
        "basic_module.4.conv": "basic_module.8",
        "basic_module.3.conv": "basic_module.6",
        "basic_module.2.conv": "basic_module.4",
        "basic_module.1.conv": "basic_module.2",
        "basic_module.0.conv": "basic_module.0",
    }

    for param_name, param in state_dict.items():
        new_param_name = param_name
        if "basic_module" in param_name:
            for k, v in mapping.items():
                new_param_name = new_param_name.replace(k, v)
            LOGGER.info(f"{param_name} -> {new_param_name}")

        new_state_dict[new_param_name] = param

    torch.save(new_state_dict, CHECKPOINT)
    return new_state_dict


def trace_e2fgvi():

    e2fgvi_nuke = torch.jit.script(E2fgviNuke())
    e2fgvi_nuke.save(MODEL_FILE)
    LOGGER.info(e2fgvi_nuke.code)
    LOGGER.info(e2fgvi_nuke.graph)
    LOGGER.info("Traced flow saved: %s", MODEL_FILE)


if __name__ == "__main__":
    # Convert mmcv state dict to torch format
    # Only run this once
    # convert_mmcv_to_torch(CHECKPOINT_MMCV)

    # Convert E2FGVI model to TorchScript
    trace_e2fgvi()
