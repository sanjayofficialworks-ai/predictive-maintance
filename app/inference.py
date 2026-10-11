import __main__
import joblib
import numpy as np
import pandas as pd
model=joblib.load("models/model.pkl")
preprocessor=joblib.load("models/preprocessor.pkl")
def predict_failure(raw_input:dict)->dict:
        df = pd.DataFrame([raw_input])
        df['Temp_diff [K]']=(df['Process temperature [K]']-df['Air temperature [K]'])
        df['Mechanical Power [W]']=(df['Torque [Nm]']*df['Rotational speed [rpm]']*2*np.pi/60)
        x_processed=preprocessor.transform(df)
        probability = float(
        model.predict_proba(x_processed)[0][1]
    )
        prediction = int(probability >= 0.45)
        return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_label": (
            "FAILURE RISK"
            if prediction == 1
            else "NORMAL"
        )
    }
# if __name__=="__main__":
#     test_input = {
#         "Type": "L",
#         "Air temperature [K]": 298.1,
#         "Process temperature [K]": 308.6,
#         "Rotational speed [rpm]": 1300,
#         "Torque [Nm]": 70,
#         "Tool wear [min]": 230
#     }

#     result = predict_failure(test_input)

# print("Input:")
# print(test_input)

# print("\nResult:")
# print(result)

    
