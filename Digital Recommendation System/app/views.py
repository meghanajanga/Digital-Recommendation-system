from django.shortcuts import render, redirect
from django.contrib import messages
# Option A: Default User
from django.contrib.auth.models import User
import os
import re
import string   # ✅ Add this line
import pandas as pd
from django.shortcuts import render, redirect
from django.http import HttpResponse

import pandas as pd
import joblib
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Global variables
df = None
x_train = x_test = y_train = y_test = model = None
labels = ['Data Science','Marketing','Business','Programming','Design']

def home(request):
    return render(request, 'home.html')

def admin_login(request):
    if request.method == 'POST':
        uname = request.POST['username']
        pwd = request.POST['password']
        if uname == 'admin' and pwd == 'admin':
            return redirect('admin_dashboard')
        else:
            messages.error(request, "Invalid credentials")
            return redirect('admin_login')
    return render(request, 'adminlogin.html')

def admin_dashboard(request):
    return render(request, 'admin_dashboard.html')

import pandas as pd
from django.shortcuts import render, redirect
from django.contrib import messages

df = None  # Global DataFrame (for demo purposes; better to use sessions or DB in production)

def upload_dataset(request):
    global df
    if request.method == 'POST' and request.FILES.get('dataset'):
        try:
            csv_file = request.FILES['dataset']
            df = pd.read_csv(csv_file)  # Directly read without saving
            messages.success(request, "Dataset uploaded successfully!")
            return redirect('admin_dashboard')  # Redirect to dashboard after upload
        except Exception as e:
            messages.error(request, f"Error reading file: {str(e)}")
            return redirect('upload')
    return render(request, 'upload.html')

import os
import re
import string
import joblib
from django.shortcuts import render
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


# Global variables
df, X, y, le = None, None, None, None

def preprocess_data(request):
    global df, X, y, le

    try:
        # --- Load dataset ---
        df = pd.read_csv("online_course_recommendation.csv")

        # --- Check for missing and duplicate values ---
        nulls = df.isnull().sum()
        dups = df.duplicated().sum()

        # --- Data cleaning ---
        df.drop_duplicates(inplace=True)
        df['description'].fillna('', inplace=True)
        df['tags'].fillna('', inplace=True)
        df['category'].fillna('Unknown', inplace=True)

        # --- Text cleaning function ---
        def clean_text(text):
            text = str(text).lower()
            text = re.sub(f"[{string.punctuation}]", " ", text)
            text = re.sub(r"\s+", " ", text).strip()
            return text

        # --- Combine and clean text columns ---
        df['text'] = (df['description'] + " " + df['tags']).apply(clean_text)

        # --- Encode labels ---
        le = LabelEncoder()
        df['category_encoded'] = le.fit_transform(df['category'])
        y = df['category_encoded']

        # --- TF-IDF Vectorization ---
        os.makedirs('model', exist_ok=True)
        vectorizer_path = 'model/tfidf_vectorizer.pkl'

        if not os.path.exists(vectorizer_path):
            vectorizer = TfidfVectorizer(stop_words='english', max_features=8000)
            X = vectorizer.fit_transform(df['text'])
            joblib.dump(vectorizer, vectorizer_path)
        else:
            vectorizer = joblib.load(vectorizer_path)
            X = vectorizer.transform(df['text'])

        # --- Train-test split ---
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # --- Train the model ---
        model = LogisticRegression(max_iter=300)
        model.fit(X_train, y_train)

        # --- Save all components ---
        joblib.dump(model, 'model/category_model.pkl')
        joblib.dump(le, 'model/label_encoder.pkl')

        # --- Prepare summary output ---
        output_summary = (
            f"✅ Model trained successfully!<br>"
            f"Null values before preprocessing: {nulls.to_dict()}<br>"
            f"Duplicate rows removed: {dups}<br>"
            f"Final dataset shape: {df.shape}<br>"
            f"Vectorizer & Model saved in 'model/' directory."
        )

        # --- Convert first 10 rows to HTML for display ---
        table_html = df.head(10).to_html(classes='table table-striped', index=False)

        return render(request, 'preprocess.html', {
            'result': output_summary,
            'table': table_html
        })

    except Exception as e:
        return render(request, 'preprocess.html', {
            'result': f"❌ Error occurred: {str(e)}",
            'table': ""
        })


def split_data(request):
    global X_train, X_test, y_train, y_test, X, y
    if X is not None and y is not None:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        return render(request, 'split.html', {'result': f"Train shape: {X_train.shape}, Test shape: {X_test.shape}"})
    else:
        return render(request, 'split.html', {'result': "Please preprocess the data first."})


import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from django.shortcuts import render

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
import os, joblib
from sklearn.linear_model import LogisticRegression

def train_model(request):
    global clf, X_train, X_test, y_train, y_test, le

    clf_path = 'model/LogisticRegression_Category.pkl'

    # Check if the train/test data exists
    if 'X_train' in globals() and X_train is not None and 'y_train' in globals() and y_train is not None:
        
        # Ensure the model directory exists
        os.makedirs(os.path.dirname(clf_path) or '.', exist_ok=True)

        # Train the model if not already saved
        if not os.path.exists(clf_path):
            clf = LogisticRegression(max_iter=300)
            clf.fit(X_train, y_train)
            joblib.dump(clf, clf_path)
            message = "Model trained and saved successfully."
        else:
            clf = joblib.load(clf_path)
            message = "Model loaded successfully."

        # Make predictions on the test set
        y_pred = clf.predict(X_test)

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred) * 100

        # ✅ Precision, Recall, F1 (weighted for multi-class)
        precision = precision_score(y_test, y_pred, average='weighted') * 100
        recall = recall_score(y_test, y_pred, average='weighted') * 100
        fscore = f1_score(y_test, y_pred, average='weighted') * 100

        # Confusion Matrix + Report
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)

        return render(request, 'train.html', {
            'message': message,
            'accuracy': f"{accuracy:.2f}%",
            'precision': f"{precision:.2f}%",
            'recall': f"{recall:.2f}%",
            'fscore': f"{fscore:.2f}%",
            'confusion_matrix': cm,
            'classification_report': report
        })
    
    else:
        return render(request, 'train.html', {
            'error': 'Please preprocess and split the data first.'
        })

    
from django.contrib.auth.models import User
from django.contrib import messages
from django.shortcuts import render, redirect

from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from .models import User  # your custom User model

from django.contrib import messages
from django.shortcuts import render, redirect
from django.contrib.auth.hashers import make_password
from .models import User

from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import make_password
from .models import User  # Custom User model

def register(request):
    if request.method == 'POST':
        # Safely get all form fields
        name = request.POST.get('name')
        email = request.POST.get('email')
        phone = request.POST.get('phone')
        city = request.POST.get('city')
        state = request.POST.get('state')
        country = request.POST.get('country')
        password = request.POST.get('password')
        confirm = request.POST.get('confirm_password')

        # Validate required fields
        if not all([name, email, phone, city, state, country, password, confirm]):
            messages.error(request, "All fields are required.")
            return redirect('register')

        if password != confirm:
            messages.error(request, "Passwords do not match.")
            return redirect('register')

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return redirect('register')

        # Save user with hashed password
        user = User(
            name=name,
            email=email,
            phone=phone,
            city=city,
            state=state,
            country=country,
            password=make_password(password)
        )
        user.save()
        messages.success(request, "Registration successful.")
        return redirect('user_login')

    return render(request, 'register.html')





from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth.hashers import make_password, check_password
from .models import User  # Your custom User model

from django.contrib.auth.hashers import check_password

def user_login(request):
    if request.method == 'POST':
        email = request.POST['email']
        password = request.POST['password']

        try:
            user = User.objects.get(email=email)
            if check_password(password, user.password):
                request.session['user_id'] = user.id
                return redirect('user_input')
            else:
                messages.error(request, "Invalid credentials")
                return redirect('user_login')
        except User.DoesNotExist:
            messages.error(request, "Invalid credentials")
            return redirect('user_login')

    return render(request, 'userlogin.html')

from django.shortcuts import render
import pandas as pd
import joblib
import os
import re, string
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# Feature columns
FEATURE_COLUMNS = ['user_id', 'course_id', 'description', 'Category', 'tags']

# Course details
course_info = {
    'Data Science': {
        'description': 'Covers machine learning, analytics, data processing, and Python skills to build data-driven solutions.',
        'skills': ['Python', 'Machine Learning', 'Analytics', 'Data Processing'],
        'career_paths': ['Data Analyst', 'Data Scientist', 'ML Engineer'],
        'resources': [
            {'type': 'Book', 'title': 'Python for Data Analysis', 'author': 'Wes McKinney'},
            {'type': 'Course', 'title': 'Machine Learning A-Z', 'platform': 'Udemy'}
        ]
    },
    'Marketing': {
        'description': 'Covers marketing strategies, analytics, digital marketing, and business promotion skills.',
        'skills': ['Digital Marketing', 'Analytics', 'Strategy', 'SEO'],
        'career_paths': ['Marketing Analyst', 'Digital Marketer', 'Brand Manager'],
        'resources': [
            {'type': 'Book', 'title': 'Marketing Management', 'author': 'Philip Kotler'},
            {'type': 'Course', 'title': 'Digital Marketing Masterclass', 'platform': 'Coursera'}
        ]
    },
    'Business': {
        'description': 'Covers business strategy, management, analytics, and leadership skills to drive business growth.',
        'skills': ['Business Strategy', 'Analytics', 'Leadership', 'Management'],
        'career_paths': ['Business Analyst', 'Project Manager', 'Operations Manager'],
        'resources': [
            {'type': 'Book', 'title': 'The Lean Startup', 'author': 'Eric Ries'},
            {'type': 'Course', 'title': 'Business Analytics Specialization', 'platform': 'Coursera'}
        ]
    },
    'Programming': {
        'description': 'Covers programming languages, software development, and problem-solving skills for coding careers.',
        'skills': ['Python', 'Java', 'Cloud', 'Problem Solving'],
        'career_paths': ['Software Developer', 'Backend Developer', 'Full Stack Developer'],
        'resources': [
            {'type': 'Book', 'title': 'Clean Code', 'author': 'Robert C. Martin'},
            {'type': 'Course', 'title': 'Java Programming Masterclass', 'platform': 'Udemy'}
        ]
    },
    'Design': {
        'description': 'Covers UX, UI, creative design, and tools for designing applications and products.',
        'skills': ['UX Design', 'UI Design', 'Creativity', 'Adobe Tools'],
        'career_paths': ['UI/UX Designer', 'Graphic Designer', 'Product Designer'],
        'resources': [
            {'type': 'Book', 'title': 'Don’t Make Me Think', 'author': 'Steve Krug'},
            {'type': 'Course', 'title': 'User Experience Design Fundamentals', 'platform': 'Skillshare'}
        ]
    }
}

# ---------------------------
# Load models once
# ---------------------------
VECTOR_PATH = 'model/TFIDF_Vectorizer.pkl'
MODEL_PATH = 'model/LogisticRegression_Category.pkl'
DATA_PATH = 'online_course_recommendation.csv'

vectorizer = joblib.load(VECTOR_PATH)
model = joblib.load(MODEL_PATH)
df_original = pd.read_csv(DATA_PATH)
le = LabelEncoder()
le.fit(df_original['category'])

# ---------------------------
# Text preprocessing
# ---------------------------
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Django view
# ---------------------------
def user_input(request):
    recommended_course = None
    recommended_courses = []
    predicted_category = None
    error = None

    if request.method == 'POST':
        description = request.POST.get('description', '')
        tags = request.POST.get('tags', '')

        if not description and not tags:
            error = "Please provide description or tags."
        else:
            try:
                input_text = clean_text(f"{description} {tags}")
                X_input = vectorizer.transform([input_text])

                pred_encoded = model.predict(X_input)[0]
                predicted_category = le.inverse_transform([pred_encoded])[0]

                if predicted_category in course_info:
                    recommended_course = course_info[predicted_category].copy()
                    recommended_course['Category'] = predicted_category

                # Cosine similarity
                df_original['text'] = (df_original['description'].fillna('') + " " + df_original['tags'].fillna('')).apply(clean_text)
                X_ref = vectorizer.transform(df_original['text'])
                sims = cosine_similarity(X_input, X_ref).flatten()
                df_original['score'] = sims

                filtered = df_original[df_original['category'] == predicted_category].copy()
                top_recs = filtered.sort_values('score', ascending=False).head(5)
                recommended_courses = top_recs[['course_id', 'category', 'score', 'description']].to_dict(orient='records')

            except Exception as e:
                error = f"Error: {str(e)}"

    return render(request, 'user_input.html', {
        'recommended_course': recommended_course,
        'recommended_courses': recommended_courses,
        'predicted_category': predicted_category,
        'error': error
    })





def logout(request):
    request.session.flush()
    return redirect('home')
