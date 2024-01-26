import logging
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT = "./release_model/E2FGVI-HQ-CVPR22.pth"

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def trace_e2fgvi():
    from model import e2fgvi_hq

    data = torch.load(CHECKPOINT, map_location=DEVICE)

    e2fgvi_model = e2fgvi_hq.InpaintGenerator()
    e2fgvi_model.to(DEVICE)
    e2fgvi_model.load_state_dict(data)
    e2fgvi_model.eval()

    LOGGER.info("E2FGVI HQ Loaded.")
    LOGGER.info(e2fgvi_model)

    class E2fgviNuke(torch.nn.Module):
        def __init__(self, n_frames: int = 12, neighbor: int = 6):
            super().__init__()
            self.n_frames = n_frames
            self.neighbor = neighbor
            self.e2fgvi = e2fgvi_model

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

            unwrapped_input = x.reshape(b, c, h, n_frames, image_width).permute(0, 3, 1, 2, 4)
            pred_imgs, _ = self.e2fgvi(unwrapped_input, neighbor)

            # Return as an wide image
            return (
                pred_imgs.unsqueeze(0)
                .permute(0, 2, 3, 1, 4)
                .reshape(1, c, h, n_frames * image_width)
            )

    model_file = "./nuke/Cattery/E2FGVI/E2FGVI.pt"
    e2fgvi_nuke = torch.jit.script(E2fgviNuke())
    e2fgvi_nuke.save(model_file)
    LOGGER.info(e2fgvi_nuke.code)
    LOGGER.info(e2fgvi_nuke.graph)
    LOGGER.info("Traced flow saved: %s", model_file)


if __name__ == "__main__":
    trace_e2fgvi()
