import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np

print("=== SENTIMENT ANALYSIS EVALUATION ===")

# Check imports
try:
    from app import SentimentAnalyzer
    print("✅ Successfully imported SentimentAnalyzer")
except ImportError as e:
    print(f"❌ Failed to import SentimentAnalyzer: {e}")
    sys.exit(1)

# Check files
required_files = ['evaluation_data.csv', 'app.py']
for file in required_files:
    if os.path.exists(file):
        print(f"✅ Found {file}")
    else:
        print(f"❌ Missing {file}")

# Load data and clean it
try:
    print("📊 Loading evaluation data...")
    df = pd.read_csv('evaluation_data.csv')
    print(f"✅ Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Show the actual data structure
    print(f"\n📋 First few rows of original data:")
    print(df.head(10))
    
    # Clean the data - handle all potential issues
    original_count = len(df)
    
    # Remove rows with completely empty labels or texts
    df = df.dropna(subset=['label', 'text'], how='all')
    
    # Fill any remaining NaN values with empty strings
    df['text'] = df['text'].fillna('').astype(str)
    df['label'] = df['label'].fillna('').astype(str)
    
    # Strip whitespace
    df['text'] = df['text'].str.strip()
    df['label'] = df['label'].str.strip()
    
    # Remove rows with empty text or label after cleaning
    df = df[(df['text'].str.len() > 0) & (df['label'].str.len() > 0)]
    
    print(f"🧹 Cleaned data: {len(df)} valid samples (removed {original_count - len(df)} invalid)")
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Show final sample distribution
    print(f"\n📋 Final Sample Distribution:")
    for label in ['Positive', 'Negative', 'Neutral']:
        count = len(df[df['label'] == label])
        print(f"  {label}: {count} samples")
        
except Exception as e:
    print(f"❌ Error loading data: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Initialize model
try:
    print("\n🤖 Initializing SentimentAnalyzer...")
    analyzer = SentimentAnalyzer()
    print("✅ Model initialized successfully")
except Exception as e:
    print(f"❌ Error initializing model: {e}")
    sys.exit(1)

# Make predictions - PROCESS ALL SAMPLES
print(f"\n🎯 Making predictions for {len(df)} samples...")
predictions = []
confidences = []
processed_count = 0

for i, (index, row) in enumerate(df.iterrows()):
    try:
        text = row['text']
        true_label = row['label']
        
        # Show progress
        if (i + 1) % 5 == 0 or (i + 1) == len(df):
            print(f"  📝 Processing sample {i+1}/{len(df)}: '{text[:40]}...'")
        
        result = analyzer.analyze_text(text)
        if result:
            pred = result['sentiment_label']
            confidence = result['confidence']
            predictions.append(pred)
            confidences.append(confidence)
            processed_count += 1
            
            # Show first few predictions in detail
            if i < 3:
                print(f"    ✅ Sample {i+1}: True='{true_label}', Pred='{pred}' (conf: {confidence:.3f})")
        else:
            predictions.append('error')
            confidences.append(0.0)
            print(f"    ⚠️  No result for sample {i+1}")
            
    except Exception as e:
        print(f"    ❌ Error predicting sample {i+1}: {e}")
        predictions.append('error')
        confidences.append(0.0)

print(f"✅ Predictions completed: {processed_count}/{len(df)} successful")

# Add predictions to dataframe
df['predicted_label'] = predictions
df['confidence'] = confidences

# Save results
try:
    df.to_csv('evaluation_results.csv', index=False)
    print(f"💾 Saved evaluation_results.csv with {len(df)} samples")
    
    # Show a preview of saved results
    print(f"\n📊 Preview of saved results:")
    results_preview = df[['text', 'label', 'predicted_label', 'confidence']].head(8)
    for i, (idx, row) in enumerate(results_preview.iterrows()):
        status = "✅" if row['label'] == row['predicted_label'] else "❌"
        print(f"  {status} '{row['text'][:30]}...' → True: {row['label']}, Pred: {row['predicted_label']}")
        
except Exception as e:
    print(f"❌ Error saving results: {e}")

# Calculate metrics (only for samples that didn't error)
valid_indices = [i for i, pred in enumerate(predictions) if pred != 'error']
if valid_indices:
    y_true = [df.iloc[i]['label'] for i in valid_indices]
    y_pred = [predictions[i] for i in valid_indices]
    
    print(f"\n📊 EVALUATION METRICS")
    print("=" * 60)
    print(f"Valid predictions: {len(valid_indices)}/{len(df)}")
    
    # Calculate accuracy manually
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / len(y_true)
    print(f"🎯 Overall Accuracy: {accuracy:.2%} ({correct_predictions}/{len(y_true)})")
    
    # Calculate metrics per class
    classes = ['Positive', 'Negative', 'Neutral']
    print("\n📈 Detailed Per-class Metrics:")
    print("-" * 50)
    
    metrics = {}
    for class_name in classes:
        true_positives = sum(1 for true, pred in zip(y_true, y_pred) if true == class_name and pred == class_name)
        false_positives = sum(1 for true, pred in zip(y_true, y_pred) if true != class_name and pred == class_name)
        false_negatives = sum(1 for true, pred in zip(y_true, y_pred) if true == class_name and pred != class_name)
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        support = sum(1 for true in y_true if true == class_name)
        
        metrics[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': support,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
        
        print(f"{class_name:>8}:")
        print(f"           Precision = {precision:.3f} ({true_positives}/{true_positives + false_positives})")
        print(f"           Recall    = {recall:.3f} ({true_positives}/{true_positives + false_negatives})")
        print(f"           F1-Score  = {f1:.3f}")
        print(f"           Support   = {support} samples")
        print()
    
    # Create confusion matrix
    print(f"🎯 Confusion Matrix:")
    print("-" * 45)
    
    confusion_data = {}
    for true_class in classes:
        confusion_data[true_class] = {}
        for pred_class in classes:
            count = sum(1 for true, pred in zip(y_true, y_pred) if true == true_class and pred == pred_class)
            confusion_data[true_class][pred_class] = count
    
    # Print matrix
    header = "True \\ Pred" + "".join([f"{c:>12}" for c in classes])
    print(header)
    print("-" * len(header))
    
    for true_class in classes:
        row = f"{true_class:>11}"
        for pred_class in classes:
            row += f"{confusion_data[true_class][pred_class]:>12}"
        print(row)
    
    # Create visual confusion matrix
    plt.figure(figsize=(10, 8))
    
    matrix = []
    for true_class in classes:
        row = []
        for pred_class in classes:
            row.append(confusion_data[true_class][pred_class])
        matrix.append(row)
    
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Number of Samples'})
    plt.title('Confusion Matrix - Sentiment Analysis\n(50 Samples Evaluation)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontweight='bold', fontsize=12)
    plt.xlabel('Predicted Label', fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("\n📈 Saved confusion_matrix.png")
    
    # Confidence statistics
    valid_confidences = [conf for i, conf in enumerate(confidences) if i in valid_indices]
    print(f"\n📊 Confidence Statistics:")
    print(f"  Average: {np.mean(valid_confidences):.3f}")
    print(f"  Std Dev: {np.std(valid_confidences):.3f}")
    print(f"  Min:     {np.min(valid_confidences):.3f}")
    print(f"  Max:     {np.max(valid_confidences):.3f}")
    
    # Misclassification analysis
    print(f"\n🔍 Misclassification Analysis:")
    print("-" * 50)
    
    misclassified = []
    for i in valid_indices:
        if df.iloc[i]['label'] != df.iloc[i]['predicted_label']:
            misclassified.append({
                'text': df.iloc[i]['text'],
                'true_label': df.iloc[i]['label'],
                'predicted_label': df.iloc[i]['predicted_label'],
                'confidence': df.iloc[i]['confidence']
            })
    
    if misclassified:
        print(f"Total misclassified: {len(misclassified)}/{len(valid_indices)} ({len(misclassified)/len(valid_indices):.1%})")
        print("\nTop misclassifications (by confidence):")
        misclassified_sorted = sorted(misclassified, key=lambda x: x['confidence'], reverse=True)[:8]
        for i, misc in enumerate(misclassified_sorted, 1):
            print(f"  {i}. '{misc['text'][:50]}...'")
            print(f"      True: {misc['true_label']} → Pred: {misc['predicted_label']} (conf: {misc['confidence']:.3f})")
    else:
        print("🎉 No misclassifications found! Perfect accuracy!")
    
else:
    print("❌ No valid predictions to evaluate!")

print("\n" + "=" * 60)
print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
print(f"📁 Results saved to: evaluation_results.csv")
print(f"📊 Visualization saved to: confusion_matrix.png") 
print(f"📈 Valid predictions: {len(valid_indices)}/{len(df)}")
print("=" * 60)