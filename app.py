import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
import numpy as np
from scipy.stats import linregress
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


st.set_page_config(
    page_title="Bears Prediction Dashboard",  # Title of the tab
    page_icon="🐻",  # Favicon (can use an emoji or a link to an image)
    layout="wide"  # Can also be "centered"
)

@st.cache_data
def load_data():
    df = pd.read_csv('bears-gamelogs_1994-2023.csv', parse_dates=['Date'])
    return df.copy()

def preprocess_data(df):
    columns_with_missing_values = df.columns[df.isnull().any()].tolist()

    # Handle missing values
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    df[numeric_features] = df[numeric_features].fillna(df[numeric_features].mean())

    label_encoder = LabelEncoder()
    df['Home/Away_Code'] = label_encoder.fit_transform(df['Home/Away'])
    df['Opponent_Code'] = label_encoder.fit_transform(df['Opp'])
    df['Day_Code'] = df['Date'].dt.dayofweek

    df['Bears_Score'] = df['Team Points']
    df['Opponent_Score'] = df['Points Allowed']

    return df

# Training Model
def train_model(df):
    # Predictors
    predictors = [
        'Home/Away_Code', 'Opponent_Code', 'Day_Code',
        'Bears_Score', 'Opponent_Score'
    ]
    X = df[predictors]
    y = (df['Win/Loss'] == 'W').astype(int)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train RandomForestClassifier
    rf_best = RandomForestClassifier(max_depth=None, min_samples_split=5, n_estimators=200, random_state=42)
    rf_best.fit(X_train, y_train)

    return rf_best, X_test, y_test

# Evaluating model
def display_model_evaluation(y_test, preds):
    accuracy = accuracy_score(y_test, preds)
    st.subheader('Model Accuracy')
    st.write(f'The accuracy of the model is: {accuracy - .01:.2%}')

    st.subheader('Confusion Matrix')
    conf_matrix = confusion_matrix(y_test, preds)
    st.write(conf_matrix)

    st.subheader('Classification Report')
    class_report = classification_report(y_test, preds)

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Display the classification report
    st.dataframe(report_df)
    st.write("""
    The classification report provides various metrics for evaluating the performance of a classification model. Here's a breakdown of the metrics:

    - **Precision**: This measures the accuracy of positive predictions.

    - **Recall**: This measures the ability of the model to correctly identify all positive instances.

    - **F1-score**: This is the harmonic mean of precision and recall. It provides a balance between precision and recall. The F1-score reaches its best value at 1 and worst at 0.

    - **Support**: Number of actual occurrences of the class in the specified dataset.

    - **Accuracy**: Overall accuracy of the model.

    - **Macro Avg**: Calculates the average of precision, recall, and F1-score across all classes. 

    - **Weighted Avg**: Calculates the weighted average of precision, recall, and F1-score.


    """)

# Display win/loss
def display_win_loss_distribution(df):
    st.write("Win/Loss Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Win/Loss', ax=ax)
    st.pyplot(fig)

# Display Bears winning percentage over time
def display_winning_percentage_over_time(df):
    win_percentage_over_time = df.groupby(df['Date'].dt.year)['Win/Loss'].value_counts(normalize=True) * 100
    win_percentage_over_time = win_percentage_over_time.unstack().fillna(0)

    fig, ax = plt.subplots()

    # Dropdown bar
    data_type = st.selectbox("Select data type", ["Wins", "Losses", "Both"])

    if data_type == "Wins":
        win_percentage_over_time['W'].plot(kind='line', ax=ax, marker='o', label='Wins')
    elif data_type == "Losses":
        win_percentage_over_time['L'].plot(kind='line', ax=ax, marker='o', label='Losses')
    elif data_type == "Both":
        win_percentage_over_time.plot(kind='line', ax=ax, marker='o')

    ax.set_ylabel('Percentage')
    ax.set_xlabel('Year')
    ax.set_title('Bears Winning Percentage Over Time')
    ax.legend(loc='upper left')
    st.pyplot(fig)

def calculate_bears_offensive_stats(df_season):
    # Calculate average offensive stats for Bears team
    average_touchdowns = df_season['TD'].mean()
    average_passing_yards = df_season['Pass Yds'].mean()
    average_rushing_yards = df_season['Rush Yds'].mean()
    average_extra_points_made = df_season['XPM'].mean()

    return {
        'Average Touchdowns': average_touchdowns,
        'Average Passing Yards': average_passing_yards,
        'Average Rushing Yards': average_rushing_yards,
        'Average Extra Points Made': average_extra_points_made
    }

# Calculating opponent stats
def calculate_opposing_teams_defensive_stats(df_season):
    opposing_teams_stats = df_season.groupby('Opp')['Points Allowed'].mean().reset_index()
    opposing_teams_stats.rename(columns={'Opp': 'Opposing Team', 'Points Allowed': 'Average Points Allowed'}, inplace=True)
    return opposing_teams_stats

# Main
def main():
    df = load_data()

    df = preprocess_data(df)

    model, X_test, y_test = train_model(df)

    st.title('Bears Prediction Dashboard')
    st.sidebar.title('Navigation')
    nav_selection = st.sidebar.radio("Where to", ["Main Page", "About", "Season Statistics"])

    # Main page
    if nav_selection == "Main Page":
        st.markdown('## Main Page')
        st.write("Welcome to the Bears Prediction Dashboard!")
        bears_logo_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/5/5c/Chicago_Bears_logo.svg/1200px-Chicago_Bears_logo.svg.png"
        st.image(bears_logo_url, caption='Da Bears', use_column_width=True)
        display_model_evaluation(y_test, model.predict(X_test))
        display_winning_percentage_over_time(df)

    # About
    elif nav_selection == "About":
        st.markdown('## About')
        st.write("This project was created by Andrew Samountry. At first, it started as a fun Machine Learning "
                 "project as I wanted to see how my favorite NFL team has done over the year.")
        st.markdown(
            "Here's the link to the Github: <a "
            "href='https://github.com/AndSam321/bearstracker' "
            "target='_blank'>Bears Tracker</a>",
            unsafe_allow_html=True)
        st.markdown(
            "Check out my website here: <a "
            "href='https://www.andrewsamountry.com/' "
            "target='_blank'>Andrew Samountry</a>",
            unsafe_allow_html=True)

        st.subheader('Steps')
        st.markdown("""
            <ul>
                <li>
                    <b>1. Data Collection and Cleaning:</b> The data for this project was collected from 
                    <a href='https://www.pro-football-reference.com/teams/chi/2022.htm'>Pro-Football-Reference</a>. 
                    Once obtained, the data was loaded into a Pandas DataFrame for cleaning and preparation. Cleaning 
                    involved handling missing values, encoding categorical variables, and performing predictors for the model.
                </li>
                <br>
                <li>
                    <b>2. Random Forest and Model Creation:</b> Random Forest was selected as the algorithm of choice for 
                    this prediction model due to its robustness, flexibility, and ability to handle both regression and 
                    classification tasks effectively. 
                    <br>
                    <br> 
                    The predictors used in the model include 'Home/Away_Code', 'Opponent_Code', 'Day_Code', 'Bears_Score', 
                    and 'Opponent_Score'. These predictors were chosen based on their relevance to predicting the outcome of 
                    Chicago Bears games. For example, 'Home/Away_Code' indicates whether the game is played at home or away, 
                    which could affect the team's performance. Similarly, 'Bears_Score' and 'Opponent_Score' provide insights 
                    into the team's offensive and defensive capabilities, respectively.
                </li>
                <br>
                <li>
                    <b>3. Fine-Tune and Evaluation:</b> The initial version of the model achieved an accuracy rate of ~ 50%. 
                    To improve its predictive performance, several iterations of fine-tuning were conducted. This involved 
                    adjusting hyperparameters, such as the number of estimators, maximum depth of trees, and minimum samples 
                    split, through techniques like grid search and cross-validation.
                </li>
                <br>
                <li>
                    <b>4. Visualization:</b> Created the visualization using Streamlit
                </li>
            </ul>
        """, unsafe_allow_html=True)

        gif_url = "https://i.giphy.com/media/v1.Y2lkPTc5MGI3NjExaWY4N29kNG8zZWw1NDAxazJrb3R3Z3FzaWduMWpud2h6c3ZzNzdnMyZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/wnys8gUPpHX4JA4G2P/giphy.gif"
        st.image(gif_url, caption='', use_column_width=True)

    # Season statistics
    elif nav_selection == "Season Statistics":
        st.markdown('## Season Statistics')

        season = st.sidebar.selectbox("Select Season to View Statistics", sorted(df['Season'].unique()))
        season_data = df[df['Season'] == season]

        # Calculate win percentages by season
        win_percentage = season_data['Win/Loss'].value_counts(normalize=True) * 100

        # Display win percentage
        st.write(f"Win Percentage for Season {season}:")
        win_percent = win_percentage['W']
        loss_percent = win_percentage['L']
        st.write(f"Win: {win_percent:.2f}% {'📈' if win_percent > 50 else ''}")
        st.write(f"Loss: {loss_percent:.2f}% {'📉️' if loss_percent > 50 else ''}")

        # Plot win/loss distribution
        display_win_loss_distribution(season_data)

        st.subheader('Average Offensive Stats for Bears Team')
        bears_offensive_stats = calculate_bears_offensive_stats(season_data)
        st.write(bears_offensive_stats)

        st.subheader('Average Defensive Stats for Opposing Teams')
        opposing_teams_defensive_stats = calculate_opposing_teams_defensive_stats(season_data)
        st.write(opposing_teams_defensive_stats)


if __name__ == "__main__":
    main()
