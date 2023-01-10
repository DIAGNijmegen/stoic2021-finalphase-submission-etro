from train import do_learning

DATA_DIR = "/input/"
ARTIFACT_DIR = "/output/"

if __name__ == "__main__":
    # Substitute do_learning for your training function.
    # It is recommended to write artifacts (e.g. model weights) to ARTIFACT_DIR during training.
    artifacts = do_learning(DATA_DIR, ARTIFACT_DIR)

    print("Training completed.")
