import joblib
import argparse
from argparse import RawTextHelpFormatter

def load(model_path):
    
    model = joblib.load(model_path)
    return model
    

def predict(model, line):
    
    X = [line]
    pred = model.predict(X)
    return pred


def main():
    
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('lyric', type=str)
    args = parser.parse_args()
    
    model = load("best_model.joblib")
    prediction = predict(model, args.lyric)
    
    print(f"Predicted Album: {prediction}")

if __name__ == "__main__":
    main()