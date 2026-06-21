import csv
import re
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# 1. Load Data, Shuffle, and Train/Test Split
# -------------------------------------------------------------------------
def clean_text(text):
    return re.findall(r'\b\w+\b', text.lower())

emails = []
labels = []

# Read the CSV file (using the updated filename 'emails.csv')
with open('emails.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    next(reader)  # Skip the header row
    for row in reader:
        if len(row) >= 2:
            emails.append(row[0])
            labels.append(1 if row[1].lower() in ['spam', '1', 'yes'] else 0)

# Convert to numpy arrays for indexing
emails = np.array(emails)
labels = np.array(labels).reshape(-1, 1)

# Set a random seed for reproducible splits
np.random.seed(42)
shuffled_indices = np.random.permutation(len(emails))
emails = emails[shuffled_indices]
labels = labels[shuffled_indices]

# Split into 80% Train and 20% Test
split_idx = int(len(emails) * 0.8)
emails_train, emails_test = emails[:split_idx], emails[split_idx:]
Y_train, Y_test = labels[:split_idx], labels[split_idx:]

# -------------------------------------------------------------------------
# 2. Build Vocabulary and Tokenize using Training Set ONLY
# -------------------------------------------------------------------------
# To prevent data leakage, we build our vocabulary strictly from the training set
all_train_words = []
for email in emails_train:
    all_train_words.extend(clean_text(email))

word_counts = Counter(all_train_words)
vocab = [word for word, _ in word_counts.most_common(3000)]
word_to_idx = {word: idx for idx, word in enumerate(vocab)}

def transform_to_features(email_list):
    m_set = len(email_list)
    n_features = len(vocab)
    X_matrix = np.zeros((m_set, n_features + 1))
    X_matrix[:, 0] = 1  # Bias column
    
    for i, email in enumerate(email_list):
        words = clean_text(email)
        for word in words:
            if word in word_to_idx:
                idx = word_to_idx[word]
                X_matrix[i, idx + 1] += 1
    return X_matrix

X_train = transform_to_features(emails_train)
X_test = transform_to_features(emails_test)

# -------------------------------------------------------------------------
# 3. Logistic Regression & Optimization Functions
# -------------------------------------------------------------------------
def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

def compute_loss(X, Y, theta, lam):
    m_samples = len(Y)
    h = sigmoid(X @ theta)
    eps = 1e-15
    cross_entropy = - (Y * np.log(h + eps) + (1 - Y) * np.log(1 - h + eps))
    # Exclude intercept theta[0] from the penalty calculation
    reg_penalty = lam * np.sum(theta[1:] ** 2)
    return np.mean(cross_entropy) + reg_penalty

def train_gradient_descent(X, Y, lam, lr=0.05, iterations=800):
    m_samples, n_features = X.shape
    theta = np.zeros((n_features, 1))
    
    for _ in range(iterations):
        h = sigmoid(X @ theta)
        grad = (1 / m_samples) * (X.T @ (h - Y))
        
        # Apply L2 regularization gradient step (skip intercept)
        reg_grad = 2 * lam * theta
        reg_grad[0] = 0
        
        grad += reg_grad
        theta -= lr * grad
        
    return theta

# -------------------------------------------------------------------------
# 4. Evaluation Metrics
# -------------------------------------------------------------------------
def evaluate_predictions(Y_true, Y_pred):
    tp = np.sum((Y_true == 1) & (Y_pred == 1))
    fp = np.sum((Y_true == 0) & (Y_pred == 1))
    fn = np.sum((Y_true == 1) & (Y_pred == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1_score

# -------------------------------------------------------------------------
# 5. Iteration Over Lambda Values and Evaluation
# -------------------------------------------------------------------------
lambda_values = [0, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1, 2, 5, 10]
train_losses = []
test_losses = []

print(f"{'Lambda':<8} | {'Train Loss':<10} | {'Test Loss':<10} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
print("-" * 68)

for lam in lambda_values:
    # Train parameters entirely on the training split
    theta = train_gradient_descent(X_train, Y_train, lam=lam, lr=0.05, iterations=800)
    
    # Track performance across both splits
    train_loss = compute_loss(X_train, Y_train, theta, lam)
    test_loss = compute_loss(X_test, Y_test, theta, lam)
    
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    
    # Generate binary target classification output on test set (threshold = 0.5)
    test_predictions = (sigmoid(X_test @ theta) >= 0.5).astype(int)
    precision, recall, f1 = evaluate_predictions(Y_test, test_predictions)
    
    print(f"{lam:<8} | {train_loss:.4f}     | {test_loss:.4f}    | {precision:.4f}    | {recall:.4f} | {f1:.4f}")

# -------------------------------------------------------------------------
# 6. Generate and Save the Plot
# -------------------------------------------------------------------------
plt.figure(figsize=(10, 6))
plt.plot(lambda_values, train_losses, marker='o', label='Training Loss', color='royalblue', linewidth=2)
plt.plot(lambda_values, test_losses, marker='s', label='Test Loss (Generalization)', color='darkorange', linewidth=2)

plt.title('Bias-Variance Analysis: Training vs. Test Loss', fontsize=14)
plt.xlabel('Regularization Strength ($\lambda$)', fontsize=12)
plt.ylabel('Evaluated Loss Score', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)

# Save the figure to your directory instead of opening a UI window
output_filename = "svm_soft_margin_evaluation.png"
plt.savefig(output_filename, dpi=300, bbox_inches='tight')
print(f"\nEvaluation plot successfully created and saved to your folder as: '{output_filename}'")