import numpy as np
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Initialize Logistic Regression model with hyperparameters.
        
        Parameters:
        -----------
        learning_rate : float
            Step size for gradient descent
        num_iterations : int
            Number of iterations for gradient descent
        """
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.weights = None
        self.bias = None
        self.costs = []
    
    def sigmoid(self, z):
        """
        Compute the sigmoid function for the input z.
        
        Parameters:
        -----------
        z : array-like
            Linear combination of features and weights
            
        Returns:
        --------
        sigmoid : array-like
            Probability values between 0 and 1
        """
        # Clip z to avoid overflow in exp(-z)
        z = np.clip(z, -500, 500)  
        return 1 / (1 + np.exp(-z))
    
    def initialize_parameters(self, n_features):
        """
        Initialize weights and bias to zeros.
        
        Parameters:
        -----------
        n_features : int
            Number of features in the dataset
        """
        self.weights = np.zeros((n_features, 1))
        self.bias = 0
    
    def compute_cost(self, y, y_pred):
        """
        Compute the log loss (binary cross-entropy).
        
        Parameters:
        -----------
        y : array-like, shape (m, 1)
            True binary labels 
        y_pred : array-like, shape (m, 1)
            Predicted probabilities
            
        Returns:
        --------
        cost : float
            The value of the cost function
        """
        m = len(y)
        
        # Add small epsilon to avoid log(0)
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Calculate the log loss
        cost = -1/m * np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))
        return cost
    
    def compute_gradients(self, X, y, y_pred):
        """
        Compute the gradients of the cost function.
        
        Parameters:
        -----------
        X : array-like, shape (m, n_features)
            Training data
        y : array-like, shape (m, 1)
            True binary labels
        y_pred : array-like, shape (m, 1)
            Predicted probabilities
            
        Returns:
        --------
        dw : array-like, shape (n_features, 1)
            Gradient of the cost with respect to weights
        db : float
            Gradient of the cost with respect to bias
        """
        m = X.shape[0]
        
        # Calculate error
        error = y_pred - y
        
        # Calculate gradients
        dw = 1/m * np.dot(X.T, error)
        db = 1/m * np.sum(error)
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update parameters using gradient descent.
        
        Parameters:
        -----------
        dw : array-like, shape (n_features, 1)
            Gradient of the cost with respect to weights
        db : float
            Gradient of the cost with respect to bias
        """
        self.weights -= self.learning_rate * dw
        self.bias -= self.learning_rate * db
    
    def fit(self, X, y):
        """
        Train the logistic regression model.
        
        Parameters:
        -----------
        X : array-like, shape (m, n_features)
            Training data
        y : array-like, shape (m,)
            Target values
        """
        # Convert y to column vector if it's not
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        # Get number of samples and features
        m, n_features = X.shape
        
        # Initialize parameters
        self.initialize_parameters(n_features)
        
        # Gradient descent optimization
        for i in range(self.num_iterations):
            # Forward pass: compute prediction
            z = np.dot(X, self.weights) + self.bias
            y_pred = self.sigmoid(z)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.costs.append(cost)
            
            # Backward pass: compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            # Print cost every 100 iterations
            if i % 100 == 0:
                print(f"Cost after iteration {i}: {cost}")
                
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (m, n_features)
            Test samples
            
        Returns:
        --------
        probabilities : array-like, shape (m, 1)
            The probability of each input being class 1
        """
        z = np.dot(X, self.weights) + self.bias
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """
        Predict class labels for samples in X.
        
        Parameters:
        -----------
        X : array-like, shape (m, n_features)
            Test samples
        threshold : float, default=0.5
            The decision threshold
            
        Returns:
        --------
        predictions : array-like, shape (m, 1)
            Predicted class labels (0 or 1)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def score(self, X, y):
        """
        Calculate the accuracy of the model.
        
        Parameters:
        -----------
        X : array-like, shape (m, n_features)
            Test samples
        y : array-like, shape (m,)
            True labels
            
        Returns:
        --------
        accuracy : float
            The fraction of correctly classified samples
        """
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
            
        predictions = self.predict(X)
        accuracy = np.mean(predictions == y)
        return accuracy
    
    def plot_decision_boundary(self, X, y):
        """
        Plot the decision boundary (only works for 2D data).
        
        Parameters:
        -----------
        X : array-like, shape (m, 2)
            Training data with 2 features
        y : array-like, shape (m, 1)
            Target values
        """
        if X.shape[1] != 2:
            print("This function only works for 2D data")
            return
        
        # Set min and max values with some margin
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        
        # Create a mesh grid
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, 0.01),
                               np.arange(x2_min, x2_max, 0.01))
        
        # Flatten and stack to create all combinations
        grid = np.c_[xx1.ravel(), xx2.ravel()]
        
        # Predict class labels for all points in the grid
        Z = self.predict(grid).reshape(xx1.shape)
        
        # Plot the contour and training examples
        plt.contourf(xx1, xx2, Z, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), edgecolors='k', marker='o', s=50)
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Logistic Regression Decision Boundary')
        plt.tight_layout()
        plt.show()
    
    def plot_cost_history(self):
        """
        Plot the cost function over iterations.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(range(len(self.costs)), self.costs)
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.title('Cost Function vs Iterations')
        plt.grid(True)
        plt.show()


# Example usage with a simple dataset
def generate_example_data(n_samples=100, n_features=2):
    """Generate a simple binary classification dataset."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_features)  # Generate random features
    
    # Create a linear decision boundary
    true_weights = np.array([1, -2]).reshape(-1, 1)
    true_bias = 0.5
    
    # Calculate linear combination
    z = np.dot(X, true_weights) + true_bias
    
    # Apply sigmoid and then threshold to get binary labels
    prob = 1 / (1 + np.exp(-z))
    y = (prob > 0.5).astype(int)
    
    return X, y

# Generate example data
X, y = generate_example_data(n_samples=200)

# Create and train the model
model = LogisticRegression(learning_rate=0.1, num_iterations=1000)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)
accuracy = model.score(X, y)
print(f"Accuracy: {accuracy:.4f}")

# Plot decision boundary
model.plot_decision_boundary(X, y)

# Plot cost history
model.plot_cost_history()