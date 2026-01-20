#!/usr/bin/env python3
'''
Create vectors for 5 different phishing email features (suspicious links, urgency words, etc.).
Use dot product to classify new emails.

NOTE: This is just simple calculation stuff to understand the concept.
NOT production code. If you deploy this to prod, that's on you buddy.
'''

# lets import the libs baby, btw be aware to have your venv.
import numpy as np

# oh shoot, I just wrote more comments than the libs that I had to import, anyways lets go to the cool stuff.

'''
just an explanation for people who need a little intuition. coding is just translating this english
that I am writing here to a good program. well first of all, I think we have these 5 phishing email
features as columns...duh (remember the notes?).
the second thing would be just to think of 5 phishing emails and randomly rate them, I am going to just randomly guess.
third thing would be to normalize them and get their cosø and just see how similar they are or whatever.
'''

# Features: [suspicious_links, urgency_words, spelling_errors, unknown_sender, requests_personal_info]
# Weights for phishing classification (I just made these up, don't @ me)
weights = np.array([0.3, 0.25, 0.15, 0.2, 0.1])

# 5 sample emails (rows) x 5 features (columns)
emails = np.array([
    [5, 8, 3, 1, 9],   # email 1 - looks sus
    [1, 2, 1, 0, 0],   # email 2 - probably fine
    [7, 9, 6, 1, 8],   # email 3 - definitely phishing
    [0, 1, 0, 0, 1],   # email 4 - safe as hell
    [4, 6, 2, 1, 5]    # email 5 - borderline
])

# New email to classify (the one we actually care about)
new_email = np.array([6, 7, 4, 1, 7])

# threshold for "yeah this is phishing" (picked 3 randomly, seems reasonable)
threshold = 3.0


def similarity(mail, new_mail):
    """calculates cosine similarity between two emails - how similar are they?"""
    dot_product = np.dot(mail, new_mail)

    # we need to also normalize (to get that sweet cosø between -1 and 1)
    magnitude_product = np.linalg.norm(mail) * np.linalg.norm(new_mail)
    return dot_product / magnitude_product


# First, let's see how similar our training emails are to the new email
print("=== Similarity scores (how much does new_email look like each training email) ===")
for i, mail in enumerate(emails):
    sim_score = similarity(mail, new_email)
    print(f"Email {i+1} similarity to new_email: {sim_score:.3f}")

print("\n" + "="*70 + "\n")

# Now let's classify emails using weighted dot product
print("=== Phishing classification (weighted dot product) ===")
for i, mail in enumerate(emails):
    phishing_score = np.dot(mail, weights)
    if phishing_score > threshold:
        print(f"Email {i+1}: PHISHING (score: {phishing_score:.2f})")
    else:
        print(f"Email {i+1}: SAFE (score: {phishing_score:.2f})")

# Don't forget to actually classify the NEW email (the whole point lol)
print("\n" + "="*70 + "\n")
new_email_score = np.dot(new_email, weights)
print(f"NEW EMAIL score: {new_email_score:.2f}")
if new_email_score > threshold:
    print("Classification: PHISHING - close the Gates soldiers!, some script kiddie tried to breach haha.")
else:
    print("Classification: SAFE - let it through")
