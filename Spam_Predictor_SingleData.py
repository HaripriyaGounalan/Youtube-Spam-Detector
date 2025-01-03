import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB, GaussianNB, ComplementNB
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.metrics import classification_report
import seaborn as sns

# Setting a random seed at beginning for consistency
seed = 17  
np.random.seed(seed)

# Step a: Get the data
# Load the dataset from a local file
filename = "Youtube03-LMFAO.csv"

path = "C:/Users/Vivek K/OneDrive/Desktop/SEM 1/COMP 237 -AI/Final Project/Final Project/3 Models (MNB GNB CNB) More Training DataSet/SingleData"
fullpath = os.path.join(path, filename)

# Initialize the Youtube data variable
youtube_dataframe = None

# Attempt to read the file with specified delimiter and error handling
try:
    # 1. Loading “Youtube03-LMFAO.csv” data into a data frame
    youtube_dataframe = pd.read_csv(fullpath, sep=",", on_bad_lines="skip")
    print("Youtube data loaded successfully.")

except Exception as e:
    print(f"Error reading {filename}: {e}")


# Step b: Initial Exploration
if youtube_dataframe is not None:

    # Displays up to 500 rows and 100 columns
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)

    # 1. Display first 3 records
    print("\n....Displaying First 3 Records....\n", youtube_dataframe.head(3))

    # 2. Display shape of the dataframe
    print("\n\n....Displaying Shape of the dataframe....\n", youtube_dataframe.shape)
    print("\n\n....Displaying Data Description...\n", youtube_dataframe.describe())  # Describe data

    # 3. Display column names, type, and missing value
    print("\n\n....Displaying Column Names, Type, and Missing Values Per Column....\n")
    youtube_dataframe.info()
    print(
        "\n\n....Displaying Missing Values per Column...\n",
        youtube_dataframe.isnull().sum(),
    )

else:
    print("Youtube data could not be loaded.")

# Step c: Data Preprocessing
try:

    # 1. Identify columns to remove due to irrelevance (e.g., COMMENT_ID, AUTHOR, and DATE are not useful for analysis)
    columns_to_drop = ["COMMENT_ID", "AUTHOR", "DATE"]
    print(f"\nIdentified Columns for Removal: {columns_to_drop}")

    # 2. Drop unnecessary columns
    youtube_dataframe = youtube_dataframe.drop(columns_to_drop, axis=1)

    # 3. Prints First 2 records after transformation
    print("\n....First 2 records after transformation....")
    print(youtube_dataframe.head(2))
    # 4. Split the features and target
    X = youtube_dataframe["CONTENT"]
    Y = youtube_dataframe["CLASS"]

    # 5. Transform Data using Count Vectorizer and TF-IDF
    count_vectorizer = CountVectorizer()
    X_counts = count_vectorizer.fit_transform(X)
    print("\nDimensions of training data:", X_counts.shape)

    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)
    print("\nTF-IDF Data Dimensions:", X_tfidf.shape)

    # 6. Shuffle the dataset using pandas.sample()
    df_shuffled = youtube_dataframe.sample(frac=1, random_state=42)

except Exception as e:
    print(f"\nError in Data Preprocessing: {e}")

# Step d: Build the model
try:

    # 1. Split the data manually (75% train, 25% test)
    train_size = int(0.75 * len(df_shuffled))
    train_X = X_tfidf[:train_size]
    train_y = Y[:train_size]
    test_X = X_tfidf[train_size:]
    test_y = Y[train_size:]

    print(f"\nTraining Set Shape: {train_X.shape}")
    print(f"\nTesting Set Shape: {test_X.shape}")

    # 2. Train the Naive Bayes model
    multinomialNBClassifier = MultinomialNB()
    multinomialNBClassifier.fit(train_X, train_y)
    print("\nModel built for Multinomial NB")

    gaussianNBClassifier = GaussianNB()
    gaussianNBClassifier.fit(train_X.toarray(), train_y)
    print("\nModel built for Gaussian NB")

    complementNBClassifier = ComplementNB()
    complementNBClassifier.fit(train_X, train_y)
    print("\nModel built for Complement NB")

    # 3. Cross-validate with 5-fold
    def cross_validation(classifier, X_data): # cross-validation method

        for test_size in np.arange(0.10, 0.51, 0.05):
            sample_size = test_size/100
            x_train_crossval, x_test_crossval, y_train_crossval, y_test_crossval = train_test_split(X_data, Y, test_size=sample_size)
            scores = cross_val_score(classifier, x_train_crossval, y_train_crossval, cv=5)

            # For each run, Print out the minimum, mean and maximum accuracy
            print(f"Test size: {test_size:.2f}, Min Accuracy: {scores.min()}, Mean: {scores.mean()}, Max Accuracy: {scores.max()}")


    print("\n\n....Cross Validation for Multinomial NB:- test size 10% to 50% increasing test sample by 5%.....\n")
    cross_validation(multinomialNBClassifier, X_tfidf) # Multinomial NB

    print("\n\n....Cross Validation for Gaussian NB:- test size 10% to 50% increasing test sample by 5%.....\n")
    cross_validation(gaussianNBClassifier, X_tfidf.toarray()) # Gaussian NB

    print("\n\n....Cross Validation for Complement NB:- test size 10% to 50% increasing test sample by 5%.....\n")
    cross_validation(complementNBClassifier, X_tfidf) # Complement NB

except Exception as e:
    print(f"Error in Building the model: {e}")


# Method to predict labels for new comments
# parameter model: 
# 1 - multinomialNBClassifier: Multinomial Naive Bayes model
# 2 - gaussianNBClassifier: Gaussian Naive Bayes model
# 3 - complementNBClassifier: Complement Naive Bayes model
def predictor(new_comments, model):
    try:
        # Vectorize all new comments at once
        new_comments_vectorized = tfidf_transformer.transform(count_vectorizer.transform(new_comments))
        new_comments_dense = new_comments_vectorized.toarray()

        # Use Gaussian Naive Bayes or Multinomial Naive Bayes based on the flag
        if model == 1:
            new_predictions = multinomialNBClassifier.predict(new_comments_dense)
        elif model == 2:
            new_predictions = gaussianNBClassifier.predict(new_comments_dense)
        elif model == 3:
            new_predictions = complementNBClassifier.predict(new_comments_dense)

        #print("\n\nComment:", new_comments, "\nPrediction:", new_predictions)

        return new_predictions
    except Exception as e:
        print(f"Error in predicting: {e}")
        return None

# Method to plot classification_report
def plot_classification_report_radial(report, model_name):
    report_dict = classification_report(test_y, report, output_dict=True)
    comments = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
    metrics = ['precision', 'recall', 'f1-score']

    # Data Preparation
    data = np.array([[report_dict[comment][metric] for metric in metrics] for comment in comments])
    data = np.concatenate((data, data[:, :1]), axis=1)  # Close the loop for radial chart
    labels = metrics + [metrics[0]]  # Close the loop for radial chart

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    for i, comment in enumerate(comments):
        ax.plot(theta, data[i], label=f'Class: {comment}')
        ax.fill(theta, data[i], alpha=0.25)

    ax.set_xticks(theta)
    ax.set_xticklabels(labels)
    ax.set_title(f'{model_name} Classification Metrics (Radial)', fontsize=14)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))
    plt.tight_layout()
    #plt.show()

def plot_classification_report_radial_combined(reports, model_name):
    ctr = 0
    fig, ax = plt.subplots(1, 3, figsize=(5, 15), dpi=100 ,subplot_kw=dict(polar=True))
    plt.subplots_adjust(wspace=0.5,hspace=0.1)
    for report in reports:
        report_dict = classification_report(test_y, report, output_dict=True)
        comments = list(report_dict.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
        metrics = ['precision', 'recall', 'f1-score']

        # Data Preparation
        data = np.array([[report_dict[comment][metric] for metric in metrics] for comment in comments])
        data = np.concatenate((data, data[:, :1]), axis=1)  # Close the loop for radial chart
        labels = metrics + [metrics[0]]  # Close the loop for radial chart

        theta = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

        for i, comment in enumerate(comments):
            ax[ctr].plot(theta, data[i], label=f'Class: {comment}')
            ax[ctr].fill(theta, data[i], alpha=0.25)

        ax[ctr].tick_params(which='major', labelsize=6)
        ax[ctr].tick_params(which='minor', labelsize=6)
        ax[ctr].set_xticks(theta)
        ax[ctr].set_xticklabels(labels, fontsize=6)
        ax[ctr].set_title(f'{model_name[ctr]} Classification Metrics (Radial)', fontsize=6)
        ax[ctr].legend(loc='upper right', bbox_to_anchor=(1.2, 1.1), prop={'size': 6})

        ctr += 1
    plt.tight_layout()
    return fig

# Method to plot side-by-side confusion matrices
def plot_confusion_matrices(cm1, cm2, cm3, model1, model2, model3, show=True):
    fig, axes = plt.subplots(1, 3, figsize=(30, 5))
 
    plt.subplots_adjust(wspace=0.7,hspace=0.1)
    sns.heatmap(cm1, annot=True, fmt="d", cmap="Blues", ax=axes[0])
    axes[0].set_title(f'Confusion Matrix - {model1}', fontsize=6)
    axes[0].set_xlabel('Predicted', fontsize=6)
    axes[0].set_ylabel('Actual', fontsize=6)

    sns.heatmap(cm2, annot=True, fmt="d", cmap="Greens", ax=axes[1])
    axes[1].set_title(f'Confusion Matrix - {model2}', fontsize=6)
    axes[1].set_xlabel('Predicted', fontsize=6)
    axes[1].set_ylabel('Actual', fontsize=6)

    sns.heatmap(cm3, annot=True, fmt="d", cmap="PuBu", ax=axes[2])
    axes[2].set_title(f'Confusion Matrix - {model3}', fontsize=6)
    axes[2].set_xlabel('Predicted', fontsize=6)
    axes[2].set_ylabel('Actual', fontsize=6)

    if show:
        plt.tight_layout()
        #plt.show()
    else:
        #plt.tight_layout()
        return fig

def plot_accuracy_comparison(a1, a2, a3, show=True):
    # Plotting accuracy comparison
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    accuracies = [a1, a2, a3]
    models = ["Multinomial NB", "Gaussian NB", "Complement NB"]

    #plt.figure(figsize=(6, 4))
    sns.barplot(x=models, y=accuracies, palette="viridis", hue=models, dodge=False, ax=ax)  
    ax.set_title("Model Accuracy Comparison")
    ax.set_ylabel("Accuracy")
    if show:
        plt.tight_layout()
        #plt.show()
    else:
        return fig

#Step e: Test the model
try:

    predictions1 = multinomialNBClassifier.predict(test_X)    # Multinomial Naive Bayes predictions
    predictions2 = gaussianNBClassifier.predict(test_X.toarray())    # Gaussian Naive Bayes predictions
    predictions3 = complementNBClassifier.predict(test_X)    # Complement Naive Bayes predictions

    # Accuracy 
    accuracy1= accuracy_score(test_y, predictions1)
    accuracy2 = accuracy_score(test_y, predictions2)
    accuracy3 = accuracy_score(test_y, predictions3)

    # confusion matrix
    cm1 = confusion_matrix(test_y, predictions1)
    cm2 = confusion_matrix(test_y, predictions2)
    cm3 = confusion_matrix(test_y, predictions3)

    # plotting Confusion Matrices
    plot_confusion_matrices(cm1, cm2, cm3, 
                            "Multinomial NB", "Gaussian NB", "Complement NB")
    
    # 2. Confusion matrix, accuracy and classification report
    print("\n\n....Mutinomial NB.....\n") # Multinomial NB
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions1))
    print(f"Accuracy: {accuracy1:.2f}")
    print("\nClassification Report:\n", classification_report(test_y, predictions1))
    plot_classification_report_radial(predictions1, "Multinomial NB")

    print("\n\n....Gaussian NB.....\n") # Gaussian NB
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions2))
    print(f"Accuracy: {accuracy2:.2f}")
    print("\nClassification Report:\n", classification_report(test_y, predictions2))
    plot_classification_report_radial(predictions2, "Gaussian NB")

    print("\n\n....Complement NB.....\n") # Gaussian NB
    print("Confusion Matrix:\n", confusion_matrix(test_y, predictions3))
    print(f"Accuracy: {accuracy2:.2f}")
    print("\nClassification Report:\n", classification_report(test_y, predictions3))
    plot_classification_report_radial(predictions3, "Complement NB")


    plot_accuracy_comparison(accuracy1, accuracy2, accuracy3)


    # 3. New comments to test (4 non-spam, 2 spam)
    new_comments = [
        ["Somehow, I end up here every single day.", 0], # Non-spam
        ["Huh, anyway check out this you[tube] channel: kobyoshi02.", 1], # Spam
        ["I knew I had to watch this song when I heard my favorite comedian was being featured.", 0], # Non-spam
        ["I love this song if you agree with me, please hit the like button.", 0], # Non-spam
        ["Let's be honest; Google algorithms did not recommend this to you. You searched for it.", 0], # Non-spam
        ["You should check my channel for Funny VIDEOS!!", 1] # Spam
    ]

    # Print the comparison of the results
    print("\n\nComparison of Predictions for New Comments:")

    new_predictions1 = predictor([comment[0] for comment in new_comments], 1) # call predictor for Multinomial NB
    new_predictions2 = predictor([comment[0] for comment in new_comments], 2) # call predictor for Gaussian NB
    new_predictions3 = predictor([comment[0] for comment in new_comments], 3) # call predictor for Complement NB
    
    # Displaying the results in a tabular format for better clarity
    print("\n{:<5} {:<90} {:<15} {:<15} {:<15} {:<15}".format('No.', 'Comment','Real', 'Multinomial NB', 'Gaussian NB', "Complement NB"))

    for i, comment in enumerate(new_comments):
        prediction1 = 'Spam' if new_predictions1[i] == 1 else 'Non-Spam'
        prediction2 = 'Spam' if new_predictions2[i] == 1 else 'Non-Spam'
        prediction3 = 'Spam' if new_predictions3[i] == 1 else 'Non-Spam'
        real = 'Spam' if comment[1] == 1 else 'Non-Spam'
        print("{:<5} {:<90} {:<15} {:<15} {:<15} {:<15}".format(i + 1, comment[0], real, prediction1, prediction2, prediction3))

    new_accuracy1 = accuracy_score([comment[1] for comment in new_comments], new_predictions1)
    new_accuracy2 = accuracy_score([comment[1] for comment in new_comments], new_predictions2)
    new_accuracy3 = accuracy_score([comment[1] for comment in new_comments], new_predictions3)

    print(f"\nMultinomial NB Accuracy for New Comments: {new_accuracy1:.2f}")
    print(f"Gaussian NB Accuracy for New Comments: {new_accuracy2:.2f}")
    print(f"Complement NB Accuracy for New Comments: {new_accuracy3:.2f}")

    plot_accuracy_comparison(new_accuracy1, new_accuracy2, new_accuracy3)


        
except Exception as e:
    print(f"Error in Testing the model: {e}")