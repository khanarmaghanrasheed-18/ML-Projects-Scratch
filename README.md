# Cricket Win Probability Predictor (From Scratch)

This project implements a Cricket Win Probability Predictor using Logistic Regression built completely from scratch using NumPy.

## Motivation

While watching cricket matches, especially during World Cups, I was always curious about the win probability percentages shown during live matches.

This project is my attempt to understand and replicate that concept using Machine Learning.

## Learning Source

This project is part of my self-learning journey through Andrew Ng’s CS229 lectures on YouTube along with his notes and problem sets.

## Features

* Data cleaning and preprocessing
* Feature scaling implemented manually
* Logistic Regression from scratch (no scikit-learn)
* Gradient Descent optimization
* Log Loss (Maximum Likelihood)
* Train/Test split for proper evaluation

## Results

* Training Accuracy: ~77%
* Test Accuracy: ~77%

## Try It Yourself

You can run the code and input your own match situation:

* Runs remaining
* Balls remaining
* Wickets remaining
* Current run rate
* Required run rate

The model will predict the probability of winning.

## Installation & Usage

### 1. Clone the repository

```bash
git clone https://github.com/your-username/cricket-win-probability-ml.git
cd cricket-win-probability-ml
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the project

```bash
python probability_meter.py
```

## Example

Enter match conditions and get output like:

Win Probability: 45.19%

## Future Improvements

* Better feature engineering
* Improve model accuracy
* Build a web interface (Streamlit)

## Contributing

Feel free to fork the repo and experiment with your own improvements.

---

If you found this interesting, consider giving it a star!

