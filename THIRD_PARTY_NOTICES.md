# Third-party notices

This repository interoperates with or reimplements concepts from external projects and publications. Their licenses and citation requirements remain independent of this repository's MIT license.

- **TotalSegmentator** — external dependency used for anatomical segmentation. The project declares Apache-2.0 for its software; individual model/task terms may differ. Check the current upstream terms before deployment.
- **PyTorch / torchvision** — model and training framework.
- **SAM / ASAM** — the recovered training code referenced `davda54/sam` (MIT). The public version in this repository is a clean implementation of the two-step adaptive SAM update used by the archived training script.
- **AdaCos** — the recovered code referenced `4uiiurz1/pytorch-adacos` (MIT; copyright Takato Kimura). The implementation here is a clean, checkpoint-compatible implementation of the AdaCos equations. Cite the AdaCos publication when methodologically appropriate.
- **ChestX-ray14** — dataset files are not distributed here. Users are responsible for following NIH dataset terms and citation requirements.

No third-party model weights or patient images are included in the repository package.
