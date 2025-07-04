import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('train.csv')

st.title("Titanic:binary classification project")
st.sidebar.title("Table of contents")
pages=["Exploration","DataVizualization","Modeling"]
page=st.sidebar.radio("Go to", pages)

if page==pages[0]:
    st.write("### Presentation of data")
    st.dataframe(df.head())
    st.write(df.shape)
    st.dataframe(df.describe())

    if st.checkbox("Show NA"):
        st.dataframe(df.isna().sum())


if page==pages[1]:
    st.write("### Data vizualization")

    fig=plt.figure()
    plt.title("Dsitribution of the Survived passengers")
    sns.countplot(x='Survived', data=df, palette='pastel')
    st.pyplot(fig)
    plt.close(fig)

    gig = plt.figure()
    sns.countplot(x = 'Sex', data = df)
    plt.title("Distribution of the passengers gender")
    st.pyplot(gig)

    fig = plt.figure()
    sns.countplot(x = 'Pclass', data = df)
    plt.title("Distribution of the passengers class")
    st.pyplot(fig)

    fig = sns.displot(x = 'Age', data = df)
    plt.title("Distribution of the passengers age")
    st.pyplot(fig)

    fig = plt.figure()
    sns.countplot(x = 'Survived', hue='Sex', data = df)
    st.pyplot(fig)
    fig = sns.catplot(x='Pclass', y='Survived', data=df, kind='point')
    st.pyplot(fig)
    fig = sns.lmplot(x='Age', y='Survived', hue="Pclass", data=df)
    st.pyplot(fig)

    fig, ax = plt.subplots()
    numeric_df = df.select_dtypes(include=['number'])  # Only numeric columns
    sns.heatmap(numeric_df.corr(), ax=ax, annot=True, cmap="coolwarm")
    st.pyplot(fig)

if page == pages[2] : 
    st.write("### Modelling")

    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)

    y = df['Survived']
    X_cat = df[['Pclass', 'Sex',  'Embarked']]
    X_num = df[['Age', 'Fare', 'SibSp', 'Parch']]

    for col in X_cat.columns:
        X_cat[col] = X_cat[col].fillna(X_cat[col].mode()[0])
    for col in X_num.columns:
        X_num[col] = X_num[col].fillna(X_num[col].median())
    X_cat_scaled = pd.get_dummies(X_cat, columns=X_cat.columns)
    X = pd.concat([X_cat_scaled, X_num], axis = 1)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train[X_num.columns] = scaler.fit_transform(X_train[X_num.columns])
    X_test[X_num.columns] = scaler.transform(X_test[X_num.columns])

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix

    def prediction(classifier):
        if classifier == 'Random Forest':
            clf = RandomForestClassifier()
        elif classifier == 'SVC':
            clf = SVC()
        elif classifier == 'Logistic Regression':
            clf = LogisticRegression()
        clf.fit(X_train, y_train)
        return clf

    def scores(clf, choice):
        if choice == 'Accuracy':
            return clf.score(X_test, y_test)
        elif choice == 'Confusion matrix':
            return confusion_matrix(y_test, clf.predict(X_test))

    choice = ['Random Forest', 'SVC', 'Logistic Regression']
    option = st.selectbox('Choice of the model', choice)
    st.write('The chosen model is :', option)

    model_filename = f"{option.replace(' ', '_')}_model.joblib"

    if os.path.exists(model_filename):
        clf = joblib.load(model_filename)
        st.write("✅ Loaded cached model from file.")
    else:
        clf = prediction(option)
        joblib.dump(clf, model_filename)
        st.write("🆕 Trained new model and saved it.")

    display = st.radio('What do you want to show ?', ('Accuracy', 'Confusion matrix'))
    if display == 'Accuracy':
        st.write(scores(clf, display))
    elif display == 'Confusion matrix':
        st.dataframe(scores(clf, display))

