from pathlib import Path

import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "data.csv"


@st.cache_resource(show_spinner="Preparing the recommendation engine...")
def load_recommender():
    """Load and vectorize the same metadata used by the notebook model."""
    books = pd.read_csv(DATA_PATH)
    columns = [
        "isbn10",
        "title",
        "authors",
        "categories",
        "thumbnail",
        "description",
        "average_rating",
        "published_year",
    ]
    books = books[columns].dropna(subset=["title", "authors", "categories", "description"])
    books = books.reset_index(drop=True)

    books["tags"] = (
        books["authors"].str.replace(" ", "", regex=False)
        + " "
        + books["categories"].str.replace(" ", "", regex=False)
        + " "
        + books["description"]
    ).str.lower()

    vectorizer = CountVectorizer(max_features=5000, stop_words="english")
    vectors = vectorizer.fit_transform(books["tags"])
    return books, vectors


def get_recommendations(selected_index, books, vectors, count=5):
    scores = cosine_similarity(vectors[selected_index], vectors).ravel()
    nearest = scores.argsort()[::-1]
    nearest = [index for index in nearest if index != selected_index][:count]
    return books.iloc[nearest]


st.set_page_config(page_title="StudyShelf", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
        .stApp { background: #f7f5ef; color: #20261e; }
        .hero { padding: 3.3rem 0 2rem; text-align: center; }
        .hero h1 { color: #183b2b; font-size: clamp(2.4rem, 5vw, 4.6rem); margin: 0; }
        .hero p { color: #687267; font-size: 1.15rem; margin: .7rem auto 0; max-width: 620px; }
        .eyebrow { color: #c66a3d; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; font-size: .78rem; }
        .book-card { background: #fffdf8; border: 1px solid #e5dfd2; border-radius: 16px; padding: 1.1rem; min-height: 270px; }
        .book-card h3 { color: #183b2b; font-size: 1.02rem; line-height: 1.25; margin: .6rem 0 .25rem; }
        .book-card p { color: #687267; font-size: .86rem; margin: 0; }
        .tag { display: inline-block; margin-top: .75rem; padding: .25rem .55rem; background: #e5f0e9; color: #276044; border-radius: 999px; font-size: .76rem; }
        div.stButton > button { background: #c66a3d; color: white; border: 0; border-radius: 8px; font-weight: 700; width: 100%; }
        div.stButton > button:hover, div.stButton > button:active, div.stButton > button:focus {
            background: #c66a3d !important; color: white !important; border: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

books, vectors = load_recommender()

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Find your next great read</div>
      <h1>StudyShelf</h1>
      <p>Choose a book you enjoyed and discover five related books to deepen your learning.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

left, center, right = st.columns([1, 3, 1])
with center:
    selected_index = st.selectbox(
        "Search the library",
        options=books.index.tolist(),
        format_func=lambda index: f"{books.at[index, 'title']} — {books.at[index, 'authors']}",
        index=None,
        placeholder="Start typing a book title...",
    )
    find_books = st.button("Find related books", disabled=selected_index is None)

if find_books and selected_index is not None:
    selected = books.iloc[selected_index]
    recommendations = get_recommendations(selected_index, books, vectors)
    st.markdown(f"## Because you selected *{selected['title']}*")
    st.caption("Recommendations are based on shared subjects, authors, and description keywords.")

    cards = st.columns(5)
    for column, (_, book) in zip(cards, recommendations.iterrows()):
        with column:
            if pd.notna(book["thumbnail"]):
                st.image(book["thumbnail"], width=130)
            else:
                st.markdown("### 📕")
            rating = f"★ {book['average_rating']:.1f}" if pd.notna(book["average_rating"]) else "No rating"
            year = str(int(book["published_year"])) if pd.notna(book["published_year"]) else "Year unknown"
            st.markdown(
                f"""<div class="book-card"><h3>{book['title']}</h3>
                <p>{book['authors']}</p><span class="tag">{rating} · {year}</span></div>""",
                unsafe_allow_html=True,
            )
else:
    st.info("Select a title above, then choose **Find related books** to get started.")
