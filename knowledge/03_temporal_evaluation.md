# Temporal Evaluation
Fraud detection is time dependent. The project uses the earliest 70 percent for training, the next 15 percent for validation, and the latest 15 percent for final testing. The validation set selects the operating threshold. The final test set is evaluated only after the model and threshold are frozen.
