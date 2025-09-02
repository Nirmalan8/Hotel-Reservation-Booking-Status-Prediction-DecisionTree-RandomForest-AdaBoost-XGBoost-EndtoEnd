# Hotel Reservation Booking Status Prediction

##  Project Overview
This project predicts the **booking status of hotel reservations** (such as confirmed, canceled, or no-show) using multiple machine learning models. The solution includes **data preprocessing, feature engineering, model building, hyperparameter tuning, and evaluation**.

The models implemented are:
- Decision Tree Classifier
- Random Forest Classifier
- AdaBoost Classifier
- XGBoost Classifier

The goal is to compare different algorithms and identify the best-performing model.

---

##  Project Structure
Hotel-Reservation-Booking-Status-Prediction-Project/
│
├── Hotel-Reservation-Booking-Status-Prediction-using-DecisionTree,RandomForest,AdaBoost,XGBoost-EndtoEnd-Project.ipynb
├── README.md
└── dataset/ (https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset)

---

##  **Technologies Used**
- **Programming Language**: Python
- **Libraries**:
  - pandas, numpy (Data manipulation)
  - matplotlib, seaborn (Data visualization)
  - scikit-learn (Machine Learning)
  - xgboost (Gradient Boosting)
- **IDE**: Jupyter Notebook

---

##  **Dataset**
The dataset contains hotel booking details such as:
- **Customer Information**: Name, age, contact
- **Booking Details**: Number of adults, children, special requests
- **Reservation Info**: Room type, meal plan, arrival date
- **Target Variable**: `booking_status` (e.g., Confirmed, Canceled, No-show)

---

##  **Workflow**
1. **Data Loading & Exploration**
   - Understand the structure and characteristics of the data
   - Handle missing values and outliers

2. **Data Preprocessing**
   - Encoding categorical variables
   - Feature scaling (if required)
   - Train-test split with stratification

3. **Model Building**
   - Train multiple models:
     - Decision Tree
     - Random Forest
     - AdaBoost
     - XGBoost

4. **Hyperparameter Tuning**
   - GridSearchCV  for optimal parameters

5. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - Confusion Matrix
   - Feature Importance Visualization

6. **Model Selection**
   - Choose the best-performing model based on metrics

---

##  **Results**
- Best model: **Random Forest** 
- Achieved accuracy: **89.81 %**
- Feature importance visualization included
- <img width="846" height="636" alt="image" src="https://github.com/user-attachments/assets/89497933-5f74-4398-ab94-f78ec0ff2788" />


---
## **Future Enhancements**

-Deploy the model using Streamlit or Flask

-Add cross-validation for robust performance

-Integrate deep learning models for comparison

---
##  **How to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/Nirmalan8/hotel-reservation-prediction.git
   cd hotel-reservation-prediction
