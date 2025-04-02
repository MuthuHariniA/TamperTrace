import torch

class WatermarkComparison:
    """Compares EditGuard with other watermarking methods."""
    
    def __init__(self):
        self.methods = ["SepMark", "PIMoG", "MBRS", "CIN", "EditGuard"]

    def compare_fidelity(self):
        """Simulates comparison of image fidelity across watermarking methods."""
        fidelity_scores = {method: torch.rand(1).item() * 10 for method in self.methods}
        return fidelity_scores

    def evaluate_perceptual_quality(self):
        """Simulates evaluation of perceptual quality like NIQE."""
        perceptual_scores = {method: torch.rand(1).item() * 10 for method in self.methods}
        return perceptual_scores

if __name__ == "__main__":
    comparator = WatermarkComparison()
    print("Fidelity Scores:", comparator.compare_fidelity())
    print("Perceptual Quality Scores:", comparator.evaluate_perceptual_quality())
