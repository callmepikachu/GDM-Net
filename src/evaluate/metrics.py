import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from typing import Dict, List, Any, Tuple
import logging


class MetricsCalculator:
    """Calculate various evaluation metrics."""
    
    def __init__(self, num_classes: int = 5):
        self.num_classes = num_classes
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def compute_accuracy(self, predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """Compute accuracy."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        return accuracy_score(labels, predictions)
    
    def compute_precision_recall_f1(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor,
        average: str = 'weighted'
    ) -> Tuple[float, float, float]:
        """Compute precision, recall, and F1 score."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, predictions, average=average, zero_division=0
        )
        
        return precision, recall, f1
    
    def compute_confusion_matrix(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> np.ndarray:
        """Compute confusion matrix."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        return confusion_matrix(labels, predictions, labels=range(self.num_classes))
    
    def compute_class_wise_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, List[float]]:
        """Compute class-wise precision, recall, and F1."""
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, predictions, average=None, zero_division=0
        )
        
        return {
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist()
        }
    
    def compute_all_metrics(
        self, 
        predictions: torch.Tensor, 
        labels: torch.Tensor
    ) -> Dict[str, Any]:
        """Compute all evaluation metrics."""
        
        # Overall metrics
        accuracy = self.compute_accuracy(predictions, labels)
        precision, recall, f1 = self.compute_precision_recall_f1(predictions, labels)
        
        # Class-wise metrics
        class_metrics = self.compute_class_wise_metrics(predictions, labels)
        
        # Confusion matrix
        cm = self.compute_confusion_matrix(predictions, labels)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'class_wise_precision': class_metrics['precision'],
            'class_wise_recall': class_metrics['recall'],
            'class_wise_f1': class_metrics['f1'],
            'class_wise_support': class_metrics['support'],
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print metrics in a formatted way."""
        
        print("\n" + "="*50)
        print("EVALUATION METRICS")
        print("="*50)
        
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        print("\nClass-wise Metrics:")
        print("-" * 30)
        for i in range(len(metrics['class_wise_f1'])):
            print(f"Class {i}:")
            print(f"  Precision: {metrics['class_wise_precision'][i]:.4f}")
            print(f"  Recall: {metrics['class_wise_recall'][i]:.4f}")
            print(f"  F1: {metrics['class_wise_f1'][i]:.4f}")
            print(f"  Support: {metrics['class_wise_support'][i]}")
        
        print("\nConfusion Matrix:")
        print("-" * 20)
        cm = np.array(metrics['confusion_matrix'])
        for i, row in enumerate(cm):
            print(f"Class {i}: {row}")
        
        print("="*50)


def compute_metrics(
    predictions: torch.Tensor, 
    labels: torch.Tensor,
    num_classes: int = 5
) -> Dict[str, Any]:
    """Convenience function to compute all metrics."""
    
    calculator = MetricsCalculator(num_classes)
    return calculator.compute_all_metrics(predictions, labels)


def compute_reasoning_metrics(
    hop_representations: List[torch.Tensor],
    entity_representations: torch.Tensor,
    ground_truth_path: List[int] = None
) -> Dict[str, float]:
    """Compute metrics specific to reasoning paths."""
    
    if not hop_representations or ground_truth_path is None:
        return {}
    
    # Extract reasoning path
    reasoning_path = []
    for hop_repr in hop_representations:
        # Find most similar entity
        similarities = torch.cosine_similarity(
            hop_repr.unsqueeze(1),
            entity_representations,
            dim=2
        )
        most_similar = similarities.argmax(dim=1)
        reasoning_path.extend(most_similar.cpu().numpy().tolist())
    
    # Compute path accuracy
    if len(reasoning_path) == len(ground_truth_path):
        path_accuracy = sum(
            p == gt for p, gt in zip(reasoning_path, ground_truth_path)
        ) / len(reasoning_path)
    else:
        path_accuracy = 0.0
    
    return {
        'reasoning_path_accuracy': path_accuracy,
        'reasoning_path_length': len(reasoning_path)
    }
