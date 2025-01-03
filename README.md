# Youtube Spam Detector
Overview:

This project involves building a spam classifier for YouTube comments using machine learning techniques. The primary focus is on evaluating and comparing the performance of three Naive Bayes classifiers:
* Multinomial Naive Bayes (MNB)
* Gaussian Naive Bayes (GNB)
* Complement Naive Bayes (CNB)

The classifiers were trained using:
1. A single dataset
2. Multiple datasets

The project evaluates the effectiveness of these classifiers in detecting spam comments on YouTube, providing insights into their accuracy, precision, recall, and overall performance across different training scenarios.

Performance After Training on a Single Dataset:
![output1](https://github.com/user-attachments/assets/364aa96a-7f34-492d-a42e-15a6f5496e2f)
![output2](https://github.com/user-attachments/assets/78051906-675a-49cd-8599-48c19f33b1f6)

Performance After Training on Multiple Datasets:
![Multiple_output1](https://github.com/user-attachments/assets/bcd24270-4b9e-4650-8f14-aeed3792bbd2)
![Multiple_output2](https://github.com/user-attachments/assets/cb4a6e66-0618-43fd-b49d-3ba0a1af67c1)

Graphical User Interface (GUI):

A Python-based GUI was developed using the PyQt framework. This interface allows users to:
1. Enter a YouTube video ID.
2. Use the trained classifiers to predict whether a comment is spam or not.
![image](https://github.com/user-attachments/assets/37f17840-ed21-462c-88b8-13975c6a7c0d)
![image](https://github.com/user-attachments/assets/81cde65d-9c46-4bfd-a95f-961e37214688)

