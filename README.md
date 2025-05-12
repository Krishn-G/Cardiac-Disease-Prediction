# Cardiac Disease Prediction

## Description
This project is a CardioVascular Disease Prediction system implemented as a PyQt5-based GUI application. It allows users to input patient data manually or load data from CSV files to predict the presence of heart disease using multiple machine learning models. The system also provides accuracy metrics for each model and helps users find nearby cardiac specialists if disease is detected.

## Features
- User-friendly GUI for data input and prediction.
- Supports manual entry of patient data or loading from CSV files.
- Utilizes multiple machine learning algorithms for prediction:
  - K-Nearest Neighbors
  - Support Vector Machine
  - Naive Bayes
  - Decision Tree
  - Logistic Regression
  - Random Forest
  - Gradient Boosting
- Displays accuracy of each model.
- Opens Google Maps to locate nearby cardiac specialists if heart disease is detected.
- Dark theme styling using qdarkstyle.
- Visualizes how much each feature contributes to model predictions
- Provides per-instance explanations by approximating the model locally with interpretable models

## Installation
1. Clone the repository.
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure you have PyQt5 installed:
   ```bash
   pip install PyQt5
   ```
4. Additional dependencies include pandas, scikit-learn, qdarkstyle.

## Usage
### Prediction
1. Run the application:
   ```bash
   python main.py
   ```
2. Use the GUI to:
   - Enter patient data manually using the input fields.
   - Or load patient data from a CSV file.
3. Select the machine learning algorithm from the dropdown.
4. Click the "Detection" button to predict heart disease.
5. View the prediction result on the GUI.
6. Click the "Accuracy" button to view accuracy metrics of all models.
7. If heart disease is detected, the app will open Google Maps to show nearby cardiac specialists.

### Model Interpretation
1. Open the SHAP_LIME.ipynb notebook in a Jupyter environment.
2. Load your model (e.g., XGBoost, RandomForest, etc.) using the provided cells.
3. Prepare your dataset (the same format used during model training).
4. Run the corresponding cells to:
    - Generate SHAP plots for a global view of feature contributions across the dataset.
    - Produce LIME explanations for individual predictions, showing which features influenced the decision and by how much.

## Dataset
The project uses heart disease datasets located in the `dataset/` directory, including:
- `heart.csv`
- `heart_semicolon.csv`
- `test.csv`

## Models
Pre-trained machine learning models are stored in the `models/` directory as pickle files:
- KNNClassifier.pkl (K-nearest Neighbors)
- SVMclassifier.pkl (Support Vector Machines)
- GNBclassifier.pkl (Naive Bayes)
- DTCclassifier.pkl (Decision Tree)
- LRclassifier.pkl (Logistic Regression)
- RFclassifier.pkl (Random Forest)
- GBclassifier.pkl (Gradient Boosting)

## Results
Accuracy results for each model are stored in the `results/` directory as pickle files and can be viewed in the GUI.

## Project Structure
- `main.py`: Main application script with GUI and prediction logic.
- `results.py`: UI dialog for displaying accuracy results.
- `MainWindow_Gui.ui`: Qt Designer UI file for the main window.
- `results.ui`: Qt Designer UI file for the results dialog.
- `dataset/`: Contains datasets used for training and testing.
- `models/`: Contains pre-trained model files.
- `results/`: Contains accuracy result files.
- `images/`: Contains images used in the GUI.
- `Training/`: Contains training notebooks and models.

## Contributing
Contributions are welcome. Please fork the repository and create a pull request with your changes.

## License
This project is licensed under the MIT License.
