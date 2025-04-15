import pandas as pd

AI = pd.read_csv('AI_Resume_Screening.csv')

AIrD = pd.DataFrame(AI)

# dont need resume id or name
AIrD = AIrD.drop(labels=['Resume_ID', 'Name'], axis=1)

# ----- education
AIrD['Education'] = AIrD['Education'].astype(str).str.strip().str.lower()

education_mapping = {
    'b.sc': 1,
    'b.tech': 2,
    'btech': 2,        # just in case
    'mba': 3,
    'm.tech': 4,
    'phd': 5
}

AIrD['Education'] = AIrD['Education'].map(education_mapping)

AIrD = AIrD.dropna(subset=['Education'])

# ----- recruiter decision: 1 hired 0 rejected
AIrD['Recruiter Decision'] = AIrD['Recruiter Decision'].astype(str).str.strip().str.lower()

AIrD['Recruiter Decision'] = AIrD['Recruiter Decision'].map({
    'hire': 1,
    'reject': 0
})

AIrD = AIrD.dropna(subset=['Recruiter Decision'])

AIrD = AIrD.reset_index(drop=True)

AIrD

AIrD.to_csv('cleaned_AI_resume_data.csv', index=False)