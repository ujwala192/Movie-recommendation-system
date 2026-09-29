# 🎬 CinemaScope AI --- Movie Recommendation System

> An intelligent, mood-aware movie discovery and recommendation platform
> built with Python and Streamlit.

## 📌 Project Overview

**CinemaScope AI** is a content-based movie recommendation system
designed to help users discover movies according to their interests and
viewing context.

The system combines **Natural Language Processing (NLP)**,
**CountVectorizer**, **Cosine Similarity**, user-defined filters, and a
**mood/activity-based recommendation engine**. Users can search and
filter movies by language, genre, rating, release year, and duration,
and can also receive recommendations based on contexts such as **Family
Night**, **Solo Evening**, and other activity modes.

The project uses IMDb movie data along with regional Telugu movie data
and provides an interactive web interface through Streamlit.

## ✨ Key Features

-   🎯 Content-based movie recommendations
-   🧠 NLP-based movie feature extraction
-   🔎 Search movies by title, genre, or keywords
-   🎭 Mood/activity-based recommendations
-   🌐 Language filtering, including Telugu content
-   ⭐ Rating-based filtering
-   📅 Release-year filtering
-   ⏱️ Duration filtering
-   🎬 Similar movie recommendations
-   🤖 AI-curated movie pick
-   📋 Detailed movie information
-   🎨 Modern, responsive Streamlit interface
-   ↩️ Back navigation and session-state management

## 🏗️ System Workflow

``` text
Movie Dataset
     ↓
Data Loading
     ↓
Data Cleaning & Preprocessing
     ↓
Feature Combination
     ↓
CountVectorizer
     ↓
Numerical Feature Vectors
     ↓
Cosine Similarity Matrix
     ↓
User Filters + Mood Selection
     ↓
Recommendation Engine
     ↓
Movie Recommendations
     ↓
Movie Details / Similar Movies / AI Pick
```

## 🧠 Recommendation Approach

### 1. Content-Based Filtering

Movie metadata such as:

-   Genre
-   Description
-   Language
-   Country

is combined into a textual feature representation.

`CountVectorizer` converts this text into numerical vectors.

### 2. Cosine Similarity

The system compares movie vectors using cosine similarity to identify
movies with similar content.

Conceptually:

``` text
Movie A → Feature Vector
Movie B → Feature Vector
       ↓
Cosine Similarity
       ↓
Similarity Score
```

Movies with higher similarity scores are considered more closely
related.

### 3. Mood-Based Recommendation

The system maps selected activities/moods to genre preferences.

For example:

``` text
Family Night
     ↓
Family / Comedy / Animation / Adventure
     ↓
Genre-weighted movie scoring
     ↓
Recommended Movies
```

The mood-based logic is combined with user filters and content relevance
to generate contextual suggestions.

## 🛠️ Technology Stack

  Technology          Purpose
  ------------------- -------------------------------------------
  Python              Core programming and recommendation logic
  Streamlit           Interactive web application
  Pandas              Data processing and manipulation
  NumPy               Numerical operations
  Scikit-learn        NLP and similarity computation
  CountVectorizer     Text feature extraction
  Cosine Similarity   Content similarity calculation
  HTML                UI structure/custom elements
  CSS                 Styling and visual design
  CSV                 Local movie data source

## 📂 Project Structure

``` text
CinemaScope-AI/
│
├── app.py
├── imdb_movies.csv
├── telugu_movies.csv
├── requirements.txt
├── README.md
│
└── assets/
    └── images/
```

> File names may be adjusted to match the actual implementation.

## ⚙️ Requirements

-   Python 3.10+
-   pip
-   Modern web browser
-   Minimum 4 GB RAM recommended
-   Approximately 1 GB free storage recommended for datasets and project
    files

## 🚀 Installation & Setup

### 1. Clone the repository

``` bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd CinemaScope-AI
```

### 2. Create a virtual environment

**Windows:**

``` bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**

``` bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

``` bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the main dependencies:

``` bash
pip install streamlit pandas numpy scikit-learn
```

### 4. Add datasets

Place the required CSV files in the project directory:

``` text
imdb_movies.csv
telugu_movies.csv
```

### 5. Run the application

``` bash
streamlit run app.py
```

The application will open in your default browser.

## 🎯 How to Use

1.  Open CinemaScope AI.
2.  Browse the available movies.
3.  Use the sidebar to select:
    -   Language
    -   Genre
    -   Rating
    -   Year range
    -   Duration
4.  Select a content mood/activity when required.
5.  Search for a movie or keyword.
6.  Open a movie's **Details** page.
7.  Explore:
    -   Similar Movies
    -   Activity/Mood Recommendations
    -   AI Pick
8.  Use the back navigation to return to the movie discovery screen.

## 🧩 Main Backend Components

The project is organized around functions/modules such as:

``` text
load_data()
clean_data()
compute_similarity_matrix()
get_content_recommendations()
get_activity_recommendations()
apply_filters()
render_search_mode()
render_details_mode()
```

### `load_data()`

Loads and combines the movie datasets.

### `clean_data()`

Cleans movie fields, handles missing values, and prepares combined
features.

### `compute_similarity_matrix()`

Uses CountVectorizer and cosine similarity to create the movie
similarity matrix.

### `get_content_recommendations()`

Returns movies similar to a selected movie.

### `get_activity_recommendations()`

Generates recommendations using activity/mood-based genre weighting.

### `apply_filters()`

Filters movies using language, genre, rating, year, and duration.

### `render_search_mode()`

Displays searchable and filtered movie results.

### `render_details_mode()`

Displays detailed movie information and recommendation sections.

## 🖥️ User Interface

The application includes:

-   Main CinemaScope AI header
-   Interactive sidebar
-   Movie search bar
-   Movie cards
-   Detailed movie view
-   Similar movie tab
-   Activity-based recommendation tab
-   AI Pick section
-   Responsive grid layout
-   Dark cinematic theme

## 📊 Example Use Case

A user wants to watch a Telugu movie released between 2015 and 2020 with
a rating of at least 6.0 and a duration below 150 minutes.

The user can select:

``` text
Language → Telugu
Year → 2015–2020
Rating → 6.0+
Duration → ≤ 150 minutes
```

The system dynamically filters the available movies.

The user can then open a movie and receive content-similar movies,
mood-based recommendations, and an AI-curated pick.

## 📈 Results

The implemented system provides:

-   Dynamic filtering without page reloads
-   Content-relevant recommendations
-   Context-aware recommendations through activity/mood mapping
-   Interactive movie exploration
-   Regional movie support
-   Modular architecture for future extensions

## 🔮 Future Scope

The project architecture can be extended with:

-   Collaborative filtering
-   Deep learning-based recommendation models
-   User profiles and viewing history
-   Advanced semantic NLP/embeddings
-   User ratings and feedback
-   TMDb/IMDb API integration
-   Larger and continuously updated movie datasets
-   More advanced personalization

## 👩‍💻 Project Information

**Project:** CinemaScope AI -- Movie Recommendation System\
**Course:** Real Time Project (CS456PC)\
**Program:** B.Tech -- Computer Science & Engineering (AI & ML)\
**Institution:** Mahatma Gandhi Institute of Technology (Autonomous),
Hyderabad\
**Academic Year:** 2024--2025

## 👥 Authors

-   **Khyati Chintha**
-   **Ujwala Addu**

## 👨‍🏫 Guidance

-   Mr. R. Srinivas --- Assistant Professor
-   Mr. K. Vikas --- Assistant Professor

## 📚 Project Focus

This project demonstrates practical application of:

``` text
Python
   +
Data Processing
   +
NLP
   +
Machine Learning
   +
Recommendation Systems
   +
Streamlit
   =
End-to-End AI Application
```

------------------------------------------------------------------------

### ⭐ Project Summary

**CinemaScope AI transforms movie discovery into a personalized
experience by combining content similarity, user-controlled filtering,
and mood/activity-aware recommendations in an interactive Streamlit
application.**
