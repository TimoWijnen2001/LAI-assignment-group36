class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Extract labels
        labels = inputs.get("labels")
        
        # Pop labels from inputs so the model doesn't try to compute its own loss
        # (This prevents the 'NotImplementedError for Float' crash in the model's forward pass)
        inputs_no_labels = {k: v for k, v in inputs.items() if k != "labels"}
        
        # Forward pass
        outputs = model(**inputs_no_labels)
        logits = outputs.get("logits")
        
        # Compute custom weighted loss
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        # Ensure labels are Long (integers)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1).long())
        
        return (loss, outputs) if return_outputs else loss