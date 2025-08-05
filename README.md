## Taylor Swift Album Classifier

Try the [TS Album Classifier](https://lyrics-app.duckdns.org/) live!

This project is a random forest classifier that predicts which Taylor Swift album a given lyric line comes from.
 
- Model & Approach
  * Data: all lyric lines from Taylor Swift’s discography, labeled by album
      + Scraped using LyricsGenius API
  * Preprocessing: cleaning (case, punctuation, etc.) and TF-IDF vectorization
  * Models evaluated (using scikit-learn unless otherwise noted):
      + Logistic regression
      + Random forest
      + Multinomial Naive Bayes
      + SGD Classifier
      + Shallow neural network (PyTorch)
  * Tuning: grid search over key parameters
  * Final model: random forest with ~37% test accuracy
- Deployment
  * App: Flask backend with a simple web UI
  * Model serving: Gunicorn WSGI server
  * Reverse proxy: Nginx
  * Hosting: AWS EC2 (with the model stored on S3)
