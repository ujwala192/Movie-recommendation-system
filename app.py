import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- App Config & Styling ---
st.set_page_config(page_title="CinemaScope AI", page_icon="🎬", layout="wide")

st.markdown("""
<style>
    .stApp { background: linear-gradient(145deg, #090d16 0%, #0f172a 50%, #111827 100%); color: #f8fafc; }
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
        padding: 1.3rem; border-radius: 14px; text-align: center; margin-bottom: 1.2rem;
        box-shadow: 0 8px 30px rgba(79, 70, 229, 0.4);
    }
    .main-title { color: #ffffff !important; font-size: 2.3rem; font-weight: 800; margin: 0; }
    .main-subtitle { color: #e0e7ff !important; font-size: 1rem; margin-top: 0.3rem; }
    
    .movie-card {
        background: #162032; border: 1px solid #29384e; border-radius: 12px;
        padding: 1rem; margin-bottom: 0.8rem; height: 190px;
        display: flex; flex-direction: column; justify-content: space-between;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-4px); border-color: #6366f1;
        box-shadow: 0 10px 25px rgba(99, 102, 241, 0.25);
    }
    .movie-title { color: #ffffff !important; font-size: 0.95rem; font-weight: 700; line-height: 1.25; margin-bottom: 0.3rem; }
    .movie-meta { color: #38bdf8 !important; font-size: 0.78rem; font-weight: 600; margin-bottom: 0.4rem; }
    .rating-badge {
        display: inline-flex; align-items: center; gap: 0.35rem; padding: 0.25rem 0.6rem;
        background: #0f172a; border: 1px solid #334155; border-radius: 6px; font-size: 0.78rem;
    }
    .star-gold { color: #fbbf24 !important; font-size: 0.9rem; }
    .rating-val { color: #f8fafc !important; font-weight: 700; }
    
    .detail-card {
        background: #162032; border: 1px solid #334155; border-radius: 14px;
        padding: 1.6rem; margin-bottom: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .badge-pill {
        display: inline-block; padding: 0.25rem 0.65rem; border-radius: 9999px;
        font-size: 0.75rem; font-weight: 600; background: #1e293b; color: #38bdf8;
        border: 1px solid #38bdf8; margin: 0.2rem;
    }
    .smart-pick {
        background: linear-gradient(135deg, #be123c 0%, #e11d48 50%, #ea580c 100%);
        color: #ffffff !important; padding: 1.4rem; border-radius: 14px; margin: 1rem 0;
        box-shadow: 0 10px 30px rgba(225, 29, 72, 0.35);
    }
    .sidebar-box {
        background: #162032; border: 1px solid #29384e; padding: 1rem;
        border-radius: 10px; margin-bottom: 1rem; border-left: 4px solid #6366f1;
    }
    .filter-info {
        background: #0f172a; border: 1px solid #334155; color: #38bdf8;
        padding: 0.3rem 0.6rem; border-radius: 6px; font-size: 0.75rem;
        font-weight: 600; text-align: center; margin-top: 0.3rem;
    }
    .stButton > button {
        background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
        color: #ffffff !important; border: none !important; border-radius: 8px !important;
        font-weight: 600 !important; font-size: 0.8rem !important; transition: 0.2s ease !important;
    }
    .stButton > button:hover {
        opacity: 0.92 !important; transform: translateY(-2px) !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4) !important;
    }
    /* Full Dark Mode Coverage for Sidebar */
    [data-testid="stSidebar"],
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    [data-testid="stSidebarContent"],
    [data-testid="stSidebarUserContent"],
    [data-testid="stSidebarNav"] {
        background-color: #0b111e !important;
        background: #0b111e !important;
        color: #f8fafc !important;
    }
    
    /* Sidebar Collapse Arrow Button */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="stSidebarCollapseButton"] button svg {
        color: #f8fafc !important;
        fill: #f8fafc !important;
        stroke: #f8fafc !important;
    }

    /* Ensure all labels, texts, and widget titles are bright, bold and clearly visible */
    label, label p,
    [data-testid="stWidgetLabel"],
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] span,
    .stTextInput label,
    .stSelectbox label,
    .stSlider label,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label p,
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #ffffff !important;
        font-size: 0.95rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.015em !important;
    }
    
    /* Sliders: tick marks, min/max values, thumb values, and numbers */
    [data-testid="stSlider"] {
        padding-top: 0.2rem !important;
        padding-bottom: 0.2rem !important;
    }
    div[data-testid="stSliderTickBarMin"],
    div[data-testid="stSliderTickBarMax"],
    div[data-testid="stSliderTickBar"] > div,
    [data-testid="stSlider"] div[data-testid="stMarkdownContainer"] p {
        color: #94a3b8 !important;
        font-weight: 700 !important;
        font-size: 0.88rem !important;
    }
    div[data-testid="stSliderThumbValue"],
    [data-testid="stSlider"] div[role="slider"],
    [data-testid="stSlider"] div[data-baseweb="slider"] div {
        color: #38bdf8 !important;
        font-weight: 800 !important;
        font-size: 0.92rem !important;
    }
    div[data-testid="stSliderThumbValue"] {
        background-color: #162032 !important;
        color: #38bdf8 !important;
        padding: 2px 7px !important;
        border-radius: 6px !important;
        border: 1px solid #38bdf8 !important;
        font-weight: 800 !important;
    }
    [data-testid="stSlider"] div {
        color: #cbd5e1 !important;
    }

    /* Search box and Text Inputs */
    div[data-testid="stTextInput"] div[data-baseweb="input"],
    div[data-testid="stTextInput"] div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-baseweb="base-input"] {
        background-color: #162032 !important;
        border: 1.5px solid #3b82f6 !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
    div[data-baseweb="input"]:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.4) !important;
        background-color: #1e293b !important;
    }
    div[data-testid="stTextInput"] input,
    div[data-baseweb="input"] input,
    input[type="text"],
    input {
        background-color: transparent !important;
        color: #ffffff !important;
        font-size: 1rem !important;
        font-weight: 500 !important;
        caret-color: #38bdf8 !important;
    }
    /* Input placeholder styling - high visibility */
    input::placeholder,
    div[data-baseweb="input"] input::placeholder,
    ::-webkit-input-placeholder,
    :-ms-input-placeholder {
        color: #94a3b8 !important;
        opacity: 1 !important;
        font-weight: 400 !important;
    }
    
    /* Dropdowns and Selectboxes */
    div[data-baseweb="select"] > div {
        background-color: #162032 !important;
        border: 1.5px solid #334155 !important;
        color: #f8fafc !important;
        border-radius: 8px !important;
    }
    div[data-baseweb="select"] span {
        color: #f8fafc !important;
    }
    div[data-baseweb="popover"], ul[role="listbox"] {
        background-color: #162032 !important;
        border: 1px solid #334155 !important;
    }
    li[role="option"] {
        color: #f8fafc !important;
    }
    li[role="option"]:hover {
        background-color: #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Configuration & Data Loading ---
ACTIVITY_GENRES = {
    "Solo Evening": {"Drama": 0.4, "Mystery": 0.3, "Thriller": 0.3},
    "Family Night": {"Family": 0.5, "Animation": 0.3, "Adventure": 0.2},
    "Date Night": {"Romance": 0.5, "Comedy": 0.3, "Drama": 0.2},
    "Friends Gathering": {"Comedy": 0.4, "Action": 0.3, "Horror": 0.3},
    "Learning Mode": {"Documentary": 0.5, "Biography": 0.3, "History": 0.2}
}

def init_state():
    defaults = {
        'selected_movie': None, 'view_mode': 'search', 'search_query': "",
        'current_activity': "Solo Evening", 'language': 'All', 'year_range': None,
        'rating_range': (6.0, 10.0), 'genre': 'All', 'duration_range': None
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

@st.cache_data
def load_data():
    try:
        imdb = pd.read_csv('imdb_movies.csv')
        cols = ['imdb_title_id', 'title', 'year', 'genre', 'duration', 'country', 'language_1', 'description', 'avg_vote', 'votes']
        df = imdb[cols].copy()
        try:
            telugu = pd.read_csv('telugu_movies.csv').rename(columns={
                'Movie': 'title', 'Year': 'year', 'Genre': 'genre', 'Overview': 'description',
                'Runtime': 'duration', 'Rating': 'avg_vote', 'No.of.Ratings': 'votes'
            })
            telugu['imdb_title_id'] = 'tl_' + telugu.index.astype(str)
            telugu['country'], telugu['language_1'] = 'India', 'te'
            df = pd.concat([df, telugu[cols]], ignore_index=True)
        except FileNotFoundError:
            pass
        
        # Clean data
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['duration'] = pd.to_numeric(df['duration'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
        df['avg_vote'] = pd.to_numeric(df['avg_vote'], errors='coerce')
        df['votes'] = pd.to_numeric(df['votes'].astype(str).str.replace(',', '', regex=True), errors='coerce')
        df['description'] = df['description'].fillna('No description available')
        df['genre'] = df['genre'].fillna('Unknown')
        df['combined_features'] = df['genre'] + ' ' + df['description'] + ' ' + df['country'].fillna('') + ' ' + df['language_1'].fillna('')
        return df.drop_duplicates(subset=['title', 'year'], keep='first')
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return pd.DataFrame()

@st.cache_resource
def get_similarity_matrix(df):
    try:
        cv = CountVectorizer(max_features=3000, stop_words='english')
        return cosine_similarity(cv.fit_transform(df['combined_features'].values.astype('U')))
    except Exception:
        return np.array([])

# --- Recommendation Helpers ---
def get_content_recommendations(title, df, sim_matrix, n=4):
    try:
        idx = df[df['title'] == title].index[0]
        scores = sorted(list(enumerate(sim_matrix[idx])), key=lambda x: x[1], reverse=True)[1:n+1]
        return df.iloc[[i[0] for i in scores]]
    except Exception:
        return pd.DataFrame()

def get_activity_recommendations(activity, df, n=4):
    if activity not in ACTIVITY_GENRES:
        return df.nlargest(n, 'avg_vote')
    df_copy = df.copy()
    df_copy['score'] = 0.0
    for genre, weight in ACTIVITY_GENRES[activity].items():
        df_copy.loc[df_copy['genre'].str.contains(genre, case=False, na=False), 'score'] += weight
    df_copy['total_score'] = (0.7 * df_copy['score']) + (0.3 * df_copy['avg_vote'] / 10)
    return df_copy.nlargest(n, 'total_score').drop_duplicates(subset=['title'])

# --- UI Components ---
def render_stars(rating):
    stars = int(round(rating / 2))
    return "★" * stars + "☆" * (5 - stars)

def render_movie_card(movie, key_prefix):
    stars = render_stars(movie['avg_vote'])
    st.markdown(f"""
    <div class="movie-card">
        <div>
            <div class="movie-title">{movie['title']}</div>
            <div class="movie-meta">📅 {int(movie['year'])} • 🎭 {str(movie['genre'])[:22]}</div>
        </div>
        <div>
            <div class="rating-badge">
                <span class="star-gold">{stars}</span>
                <span class="rating-val">{movie['avg_vote']:.1f}/10</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("🔍 Details", key=f"{key_prefix}_{movie.name}", use_container_width=True):
        st.session_state.selected_movie = movie['title']
        st.session_state.view_mode = 'details'
        st.rerun()

def render_sidebar(df):
    with st.sidebar:
        st.markdown('<div class="sidebar-box"><h3 style="color:#ffffff; margin:0; font-size:1.1rem; font-weight:700;">🔧 Filter Movies</h3></div>', unsafe_allow_html=True)
        langs = {'All': 'All Languages', 'en': 'English', 'te': 'Telugu'}
        lang = st.selectbox("🗣 Language", list(langs.keys()), format_func=lambda x: langs[x])
        
        # Duration Range
        d_min, d_max = max(40, int(df['duration'].min())), min(300, int(df['duration'].max()))
        if not st.session_state.duration_range:
            st.session_state.duration_range = (d_min, d_max)
        dur = st.slider("⏱ Duration (mins)", d_min, d_max, st.session_state.duration_range, step=5)
        st.markdown(f'<div class="filter-info">Range: {dur[0]} - {dur[1]} min</div>', unsafe_allow_html=True)
        st.session_state.duration_range = dur

        # Rating Range
        rat = st.slider("⭐ Rating", 1.0, 10.0, st.session_state.rating_range, step=0.1)
        st.markdown(f'<div class="filter-info">Rating: {rat[0]:.1f} to {rat[1]:.1f} ⭐</div>', unsafe_allow_html=True)
        st.session_state.rating_range = rat

        # Year Range
        y_min, y_max = int(df['year'].min()), int(df['year'].max())
        if not st.session_state.year_range:
            st.session_state.year_range = (max(1990, y_min), y_max)
        yr = st.slider("📅 Year", y_min, y_max, st.session_state.year_range, step=1)
        st.markdown(f'<div class="filter-info">Years: {yr[0]} - {yr[1]}</div>', unsafe_allow_html=True)
        st.session_state.year_range = yr

        # Genre Selection
        all_genres = sorted({g.strip() for sublist in df['genre'].dropna().str.split(',') for g in sublist})
        genre = st.selectbox("🎭 Genre", ['All'] + all_genres, index=(['All'] + all_genres).index(st.session_state.genre) if st.session_state.genre in (['All'] + all_genres) else 0)
        st.session_state.genre = genre

        # Content Mood
        st.markdown('<div class="sidebar-box"><h3 style="color:#ffffff; margin:0; font-size:1.1rem; font-weight:700;">🎯 Content Mood</h3></div>', unsafe_allow_html=True)
        activity = st.selectbox("Select Mood Context", list(ACTIVITY_GENRES.keys()), index=list(ACTIVITY_GENRES.keys()).index(st.session_state.current_activity))
        st.session_state.current_activity = activity

        # Reset button
        if st.button("🔄 Reset All Filters", use_container_width=True):
            st.session_state.duration_range = (d_min, d_max)
            st.session_state.rating_range = (6.0, 10.0)
            st.session_state.year_range = (max(1990, y_min), y_max)
            st.session_state.genre = 'All'
            st.session_state.search_query = ""
            st.rerun()

    return activity, lang, yr, rat, genre, dur

# --- Main App Views ---
def show_search_view(df, filters):
    activity, lang, yr, rat, genre, dur = filters
    st.markdown("### 🔍 Discover Movies")
    query = st.text_input("Search movie title, keywords...", value=st.session_state.search_query, placeholder="e.g. Inception, Avatar, Bahubali...")
    st.session_state.search_query = query

    filtered = df[
        (df['year'].between(yr[0], yr[1])) &
        (df['avg_vote'].between(rat[0], rat[1])) &
        (df['duration'].between(dur[0], dur[1]))
    ]
    if lang != 'All':
        filtered = filtered[filtered['language_1'] == lang]
    if genre != 'All':
        filtered = filtered[filtered['genre'].str.contains(genre, na=False)]

    if query.strip():
        matches = filtered[filtered['title'].str.contains(query, case=False, na=False)]
        display_df = matches.head(20) if not matches.empty else filtered.nlargest(20, 'avg_vote')
        if matches.empty:
            st.warning(f"No direct matches found for '{query}'. Showing top rated alternatives.")
        else:
            st.success(f"Found {len(matches)} movie(s) matching '{query}'")
    else:
        display_df = filtered.nlargest(20, 'avg_vote')

    if display_df.empty:
        st.info("No movies match your filter criteria. Try expanding the ranges.")
    else:
        cols = st.columns(4)
        for i, (_, movie) in enumerate(display_df.iterrows()):
            with cols[i % 4]:
                render_movie_card(movie, f"s_{i}")

def show_details_view(df, sim_matrix, activity):
    if not st.session_state.selected_movie:
        st.session_state.view_mode = 'search'
        st.rerun()
        return

    movie = df[df['title'] == st.session_state.selected_movie].iloc[0]
    
    if st.button("⬅️ Back to Discovery"):
        st.session_state.view_mode = 'search'
        st.session_state.selected_movie = None
        st.rerun()

    dur_str = f"{int(movie['duration'])} min" if pd.notna(movie['duration']) else "N/A"
    lang_str = {'en': 'English', 'te': 'Telugu'}.get(movie['language_1'], str(movie['language_1']))
    votes_str = f"{int(movie['votes']):,}" if pd.notna(movie['votes']) else "N/A"
    stars = render_stars(movie['avg_vote'])

    st.markdown(f"""
    <div class="detail-card">
        <h1 style="color:#ffffff; font-size:2rem; margin:0 0 0.5rem 0;">🎬 {movie['title']} ({int(movie['year'])})</h1>
        <div style="margin-bottom: 0.8rem;">
            <span class="badge-pill">🎭 {movie['genre']}</span>
            <span class="badge-pill">⏱ {dur_str}</span>
            <span class="badge-pill">🌍 {movie['country']}</span>
            <span class="badge-pill">🗣 {lang_str}</span>
        </div>
        <div style="display: flex; align-items: center; gap: 0.8rem; margin: 0.8rem 0;">
            <div class="rating-badge" style="padding: 0.35rem 0.8rem; font-size: 0.95rem;">
                <span class="star-gold" style="font-size: 1.1rem;">{stars}</span>
                <span class="rating-val">{movie['avg_vote']:.1f}/10</span>
                <span style="color: #94a3b8; font-size: 0.8rem; margin-left: 0.3rem;">({votes_str} votes)</span>
            </div>
        </div>
        <h4 style="color:#e2e8f0; margin-top:1rem; margin-bottom: 0.3rem;">📖 Overview</h4>
        <p style="color:#cbd5e1; line-height:1.6; font-size:0.95rem;">{movie['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎯 Recommendations for You")
    tab1, tab2, tab3 = st.tabs(["🎬 Similar Movies", f"🎭 {activity} Mood", "🤖 AI Curated Pick"])

    with tab1:
        similar = get_content_recommendations(st.session_state.selected_movie, df, sim_matrix, 4)
        if not similar.empty:
            cols = st.columns(4)
            for idx, (_, m) in enumerate(similar.iterrows()):
                with cols[idx]:
                    render_movie_card(m, f"sim_{idx}")
        else:
            st.info("No similar movies found.")

    with tab2:
        activity_movies = get_activity_recommendations(activity, df, 4)
        if not activity_movies.empty:
            cols = st.columns(4)
            for idx, (_, m) in enumerate(activity_movies.iterrows()):
                with cols[idx]:
                    render_movie_card(m, f"act_{idx}")

    with tab3:
        top_similar = get_content_recommendations(st.session_state.selected_movie, df, sim_matrix, 8)
        top_activity = get_activity_recommendations(activity, df, 8)
        pool = pd.concat([top_similar, top_activity]).drop_duplicates(subset=['title'])
        if not pool.empty:
            pick = pool.nlargest(1, 'avg_vote').iloc[0]
            st.markdown(f"""
            <div class="smart-pick">
                <h3 style="margin:0 0 0.5rem 0; font-size:1.35rem; color:#ffffff;">✨ Top Match: {pick['title']} ({int(pick['year'])})</h3>
                <p style="margin:0 0 0.6rem 0; color:#ffedd5; font-size:0.9rem;">⭐ <b>{pick['avg_vote']}/10</b> | 🎭 <b>{pick['genre']}</b></p>
                <p style="margin:0; font-size:0.9rem; line-height:1.5; color:#ffffff;">{str(pick['description'])[:200]}...</p>
            </div>
            """, unsafe_allow_html=True)
            if st.button("🚀 Explore This Pick", use_container_width=True):
                st.session_state.selected_movie = pick['title']
                st.rerun()

def main():
    init_state()
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🎬 CinemaScope AI</h1>
        <p class="main-subtitle">Smart, Mood-Aware Movie Discovery & Recommendation Platform</p>
    </div>
    """, unsafe_allow_html=True)

    df = load_data()
    if df.empty:
        st.warning("⚠️ Movie database could not be loaded. Please ensure dataset CSVs are present.")
        return

    sim_matrix = get_similarity_matrix(df)
    filters = render_sidebar(df)

    if st.session_state.view_mode == 'search':
        show_search_view(df, filters)
    else:
        show_details_view(df, sim_matrix, filters[0])

if __name__ == "__main__":
    main()
