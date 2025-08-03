# Predictive Maintenance Model

## Overview
This project implements a predictive maintenance model using a Decision Tree Classifier to predict machine failure types based on sensor data. The model is trained on a dataset (`predictive_maintenance.csv`) and deployed using IBM Watson Machine Learning. The code is written in Python and uses various libraries for data preprocessing, visualization, and machine learning.

## Dataset
The dataset (`predictive_maintenance.csv`) contains the following columns:
- **UDI**: Unique identifier for each record
- **Product ID**: Product identifier
- **Type**: Machine type (L, M, or H)
- **Air temperature [K]**: Air temperature in Kelvin
- **Process temperature [K]**: Process temperature in Kelvin
- **Rotational speed [rpm]**: Rotational speed in revolutions per minute
- **Torque [Nm]**: Torque in Newton-meters
- **Tool wear [min]**: Tool wear in minutes
- **Target**: Binary target indicating failure (0 or 1)
- **Failure Type**: Type of failure (e.g., No Failure, Overstrain Failure, etc.)

## Prerequisites
- Python 3.11 (runtime-24.1)
- Required libraries (install using the command below):
  ```bash
  pip install scikit-learn==1.4.2 pandas==2.1.4 numpy==1.26.4 matplotlib==3.8.4 seaborn==0.13.2 imbalanced-learn==0.12.3 ibm-watson-machine-learning==1.0.360
  ```
- IBM Cloud Object Storage credentials for accessing the dataset
- IBM Watson Machine Learning account and API key for model deployment

## Project Structure
- **Predictive.ipynb**: Jupyter Notebook containing the complete code for data loading, preprocessing, model training, evaluation, and deployment.
- **predictive_maintenance.csv**: Dataset used for training and testing the model (stored in IBM Cloud Object Storage).

## Setup and Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Install Dependencies**:
   Ensure Python 3.11 is installed, then run:
   ```bash
   pip install -r requirements.txt
   ```
   Alternatively, install the specific versions listed in the prerequisites.

3. **Configure IBM Cloud Object Storage**:
   Update the `Predictive.ipynb` notebook with your IBM Cloud Object Storage credentials:
   - `ibm_api_key_id`
   - `bucket`
   - `object_key`
   - `endpoint_url`

4. **Configure IBM Watson Machine Learning**:
   Update the notebook with your Watson Machine Learning credentials:
   - `url`
   - `apikey`
   - `default_project` ID

## Usage
1. **Run the Notebook**:
   Open `Predictive.ipynb` in a Jupyter environment and execute the cells sequentially. The notebook:
   - Loads the dataset from IBM Cloud Object Storage.
   - Preprocesses the data (encoding categorical variables, scaling numerical features, and handling imbalanced classes using SMOTE).
   - Trains a Decision Tree Classifier.
   - Evaluates the model using accuracy, classification report, and confusion matrix.
   - Saves the model to IBM Watson Machine Learning.
   - Provides a prediction function for new data points.

2. **Example Prediction**:
   Use the `predict_failure` function to predict the failure type for a new data point. Example:
   ```python
   sample_data = {
       'Type': 'L',
       'Air temperature [K]': 298.9,
       'Process temperature [K]': 309.1,
       'Rotational speed [rpm]': 2861,
       'Torque [Nm]': 4.6,
       'Tool wear [min]': 143
   }
   print(f"Predicted Failure Type: {predict_failure(sample_data)}")
   ```

## Model Details
- **Algorithm**: Decision Tree Classifier
- **Preprocessing**:
  - Categorical encoding using `LabelEncoder` for `Type` and `Failure Type`.
  - Feature scaling using `StandardScaler` for numerical features.
  - Class imbalance handling using SMOTE.
- **Evaluation Metrics**:
  - Accuracy
  - Classification Report (precision, recall, F1-score)
  - Confusion Matrix (visualized and saved as an image)
- **Deployment**: The model is stored in IBM Watson Machine Learning for production use.

## Results
- The model achieves an accuracy of approximately 0.93 on the test set.
- A confusion matrix is generated and saved for detailed performance analysis.
- The model can predict failure types such as "No Failure," "Overstrain Failure," etc.

## Notes
- Ensure the Python environment is set to version 3.11 to avoid compatibility issues.
- Verify that the IBM Watson Machine Learning service is enabled and the API key is valid.
- The dataset must be accessible in IBM Cloud Object Storage with the provided credentials.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or issues, please open an issue on the GitHub repository or contact the project maintainer.