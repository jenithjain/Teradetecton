import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
import torch
from torch.utils.data import DataLoader

from terratrack import ChangeDetectionDataset, ChangeDetector, load_image_pair, load_change_mask

class ChangeDetectionEvaluator:
    """
    Class for evaluating change detection models.
    """
    def __init__(self, dataset_path, labels_path):
        """
        Initialize the evaluator.
        
        Args:
            dataset_path (str): Path to the dataset
            labels_path (str): Path to the labels
        """
        self.dataset_path = dataset_path
        self.labels_path = labels_path
    
    def evaluate_model(self, model, test_locations, thresholds=None, batch_size=4):
        """
        Evaluate a change detection model on test locations.
        
        Args:
            model (ChangeDetector): Change detection model
            test_locations (list): List of test location names
            thresholds (list, optional): List of thresholds to evaluate
            batch_size (int): Batch size for evaluation
            
        Returns:
            dict: Evaluation results
        """
        if thresholds is None:
            thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        # Create dataset and dataloader
        test_dataset = ChangeDetectionDataset(test_locations, transform=model.transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        # Initialize results
        results = {
            'thresholds': thresholds,
            'precision': [],
            'recall': [],
            'f1': [],
            'accuracy': [],
            'per_location': {}
        }
        
        # Evaluate for each threshold
        for threshold in thresholds:
            print(f"Evaluating with threshold {threshold}...")
            
            # Initialize metrics
            all_preds = []
            all_masks = []
            location_results = {}
            
            # Evaluate on each batch
            for batch in test_loader:
                images1 = batch['image1'].to(model.device)
                images2 = batch['image2'].to(model.device)
                masks = batch['mask'].cpu().numpy()
                locations = batch['location']
                
                # Make predictions
                with torch.no_grad():
                    preds = model.predict_batch(images1, images2).cpu().numpy()
                
                # Apply threshold
                preds_binary = (preds > threshold).astype(np.uint8)
                
                # Store predictions and masks
                all_preds.extend(preds_binary.flatten())
                all_masks.extend(masks.flatten())
                
                # Calculate per-location metrics
                for i, location in enumerate(locations):
                    pred = preds_binary[i].flatten()
                    mask = masks[i].flatten()
                    
                    # Calculate metrics
                    precision, recall, f1, _ = precision_recall_fscore_support(
                        mask, pred, average='binary', zero_division=0)
                    accuracy = accuracy_score(mask, pred)
                    
                    # Store results
                    location_results[location] = {
                        'precision': precision,
                        'recall': recall,
                        'f1': f1,
                        'accuracy': accuracy
                    }
            
            # Calculate overall metrics
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_masks, all_preds, average='binary', zero_division=0)
            accuracy = accuracy_score(all_masks, all_preds)
            
            # Store results
            results['precision'].append(precision)
            results['recall'].append(recall)
            results['f1'].append(f1)
            results['accuracy'].append(accuracy)
            results['per_location'][threshold] = location_results
        
        return results
    
    def evaluate_single_image_pair(self, model, location, threshold=0.5):
        """
        Evaluate a model on a single image pair.
        
        Args:
            model (ChangeDetector): Change detection model
            location (str): Location name
            threshold (float): Threshold for binary prediction
            
        Returns:
            dict: Evaluation results
        """
        # Load images and mask
        img1, img2 = load_image_pair(location)
        mask = load_change_mask(location)
        
        if img1 is None or img2 is None or mask is None:
            print(f"Failed to load data for {location}")
            return None
        
        # Make prediction
        pred_prob = model.predict(img1, img2)
        pred_binary = (pred_prob > threshold).astype(np.uint8)
        
        # Calculate metrics
        precision, recall, f1, _ = precision_recall_fscore_support(
            mask.flatten(), pred_binary.flatten(), average='binary', zero_division=0)
        accuracy = accuracy_score(mask.flatten(), pred_binary.flatten())
        
        # Calculate confusion matrix
        cm = confusion_matrix(mask.flatten(), pred_binary.flatten())
        
        # Store results
        results = {
            'location': location,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'prediction_prob': pred_prob,
            'prediction_binary': pred_binary,
            'ground_truth': mask
        }
        
        return results
    
    def plot_precision_recall_curve(self, results, save_path=None):
        """
        Plot precision-recall curve from evaluation results.
        
        Args:
            results (dict): Evaluation results
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot precision-recall curve
        ax.plot(results['recall'], results['precision'], 'o-', label='Precision-Recall curve')
        
        # Add threshold annotations
        for i, threshold in enumerate(results['thresholds']):
            ax.annotate(f"{threshold:.1f}", 
                       (results['recall'][i], results['precision'][i]),
                       textcoords="offset points",
                       xytext=(0, 10),
                       ha='center')
        
        # Add F1 curve
        ax2 = ax.twinx()
        ax2.plot(results['thresholds'], results['f1'], 'r--', label='F1 Score')
        
        # Set labels and title
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax2.set_ylabel('F1 Score')
        ax.set_title('Precision-Recall Curve')
        
        # Set limits
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.0])
        
        # Add legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='lower left')
        
        # Add grid
        ax.grid(True)
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            cm (numpy.ndarray): Confusion matrix
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Figure object
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        
        # Set labels
        classes = ['No Change', 'Change']
        ax.set(xticks=np.arange(cm.shape[1]),
              yticks=np.arange(cm.shape[0]),
              xticklabels=classes, yticklabels=classes,
              title='Confusion Matrix',
              ylabel='True label',
              xlabel='Predicted label')
        
        # Rotate tick labels and set alignment
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Loop over data dimensions and create text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        fig.tight_layout()
        
        # Save plot if path is provided
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        
        return fig
    
    def plot_evaluation_results(self, results, save_dir=None):
        """
        Plot evaluation results.
        
        Args:
            results (dict): Evaluation results
            save_dir (str, optional): Directory to save plots
            
        Returns:
            list: Paths to saved plots
        """
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        
        saved_paths = []
        
        # Plot precision-recall curve
        pr_curve = self.plot_precision_recall_curve(results)
        if save_dir:
            pr_path = os.path.join(save_dir, 'precision_recall_curve.png')
            pr_curve.savefig(pr_path, bbox_inches='tight', dpi=300)
            saved_paths.append(pr_path)
        
        # Plot F1 score vs threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results['thresholds'], results['f1'], 'o-')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1 Score')
        ax.set_title('F1 Score vs Threshold')
        ax.grid(True)
        
        if save_dir:
            f1_path = os.path.join(save_dir, 'f1_vs_threshold.png')
            fig.savefig(f1_path, bbox_inches='tight', dpi=300)
            saved_paths.append(f1_path)
        
        # Plot accuracy vs threshold
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(results['thresholds'], results['accuracy'], 'o-')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Threshold')
        ax.grid(True)
        
        if save_dir:
            acc_path = os.path.join(save_dir, 'accuracy_vs_threshold.png')
            fig.savefig(acc_path, bbox_inches='tight', dpi=300)
            saved_paths.append(acc_path)
        
        # Plot per-location F1 scores
        best_threshold_idx = np.argmax(results['f1'])
        best_threshold = results['thresholds'][best_threshold_idx]
        
        location_f1 = []
        location_names = []
        
        for location, metrics in results['per_location'][best_threshold].items():
            location_names.append(location)
            location_f1.append(metrics['f1'])
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(location_names, location_f1)
        ax.set_xlabel('Location')
        ax.set_ylabel('F1 Score')
        ax.set_title(f'F1 Score per Location (Threshold = {best_threshold:.1f})')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_dir:
            loc_path = os.path.join(save_dir, 'f1_per_location.png')
            fig.savefig(loc_path, bbox_inches='tight', dpi=300)
            saved_paths.append(loc_path)
        
        return saved_paths
    
    def generate_evaluation_report(self, results, save_path=None):
        """
        Generate an evaluation report.
        
        Args:
            results (dict): Evaluation results
            save_path (str, optional): Path to save the report
            
        Returns:
            str: Report text
        """
        # Find best threshold based on F1 score
        best_idx = np.argmax(results['f1'])
        best_threshold = results['thresholds'][best_idx]
        best_precision = results['precision'][best_idx]
        best_recall = results['recall'][best_idx]
        best_f1 = results['f1'][best_idx]
        best_accuracy = results['accuracy'][best_idx]
        
        # Generate report
        report = f"Change Detection Evaluation Report\n"
        report += f"================================\n\n"
        report += f"Best Threshold: {best_threshold:.2f}\n"
        report += f"Best F1 Score: {best_f1:.4f}\n"
        report += f"Precision: {best_precision:.4f}\n"
        report += f"Recall: {best_recall:.4f}\n"
        report += f"Accuracy: {best_accuracy:.4f}\n\n"
        
        report += f"Per-Location Results (Threshold = {best_threshold:.2f}):\n"
        report += f"-------------------------------------------\n"
        
        # Add per-location results
        location_results = results['per_location'][best_threshold]
        for location, metrics in location_results.items():
            report += f"Location: {location}\n"
            report += f"  F1 Score: {metrics['f1']:.4f}\n"
            report += f"  Precision: {metrics['precision']:.4f}\n"
            report += f"  Recall: {metrics['recall']:.4f}\n"
            report += f"  Accuracy: {metrics['accuracy']:.4f}\n\n"
        
        # Save report if path is provided
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
        
        return report

# Main function for testing
def main():
    # Set paths
    dataset_path = 'd:\\tera\\Onera Satellite Change Detection dataset - Images'
    labels_path = 'd:\\tera\\Onera Satellite Change Detection dataset - Test Labels'
    
    # Get test locations
    with open(os.path.join(dataset_path, 'test.txt'), 'r') as f:
        test_locations = f.read().strip().split(',')
    
    # Initialize model
    detector = ChangeDetector()
    
    # Initialize evaluator
    evaluator = ChangeDetectionEvaluator(dataset_path, labels_path)
    
    # Evaluate on a single location
    location = 'dubai'  # Change to any available test location
    results_single = evaluator.evaluate_single_image_pair(detector, location)
    
    if results_single:
        # Plot confusion matrix
        evaluator.plot_confusion_matrix(results_single['confusion_matrix'], 'confusion_matrix.png')
        
        # Print metrics
        print(f"Evaluation results for {location}:")
        print(f"Precision: {results_single['precision']:.4f}")
        print(f"Recall: {results_single['recall']:.4f}")
        print(f"F1 Score: {results_single['f1']:.4f}")
        print(f"Accuracy: {results_single['accuracy']:.4f}")
    
    # Evaluate on all test locations (commented out as it can be time-consuming)
    # results = evaluator.evaluate_model(detector, test_locations[:2])  # Use a subset for testing
    # evaluator.plot_evaluation_results(results, 'evaluation_results')
    # report = evaluator.generate_evaluation_report(results, 'evaluation_report.txt')
    # print(report)

if __name__ == "__main__":
    main()