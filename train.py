import joblib
import smtplib
import getpass
import os
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from custom_model import LightGBDT
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from preprocessing import FeatureHashing

# Login gmail account
username = input('Input your gmail username: ')
password = getpass.getpass(prompt='Input your gmail password: ')

# Preprocessing
print('Preprocessing...')

encoder = FeatureHashing(
    train_set='dataset/train.csv',
    test_set='dataset/test.csv'
)

X, Y = encoder.training_set()
print(f'X.shape={X.shape}, Y.shape={Y.shape}')

# Split training set and testing set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
print(f'X_train.shape={X_train.shape}, X_test.shape={X_test.shape}, Y_train.shape={Y_train.shape}, Y_test.shape={Y_test.shape}')
del X
del Y

print('Training model...')
params = {
    'n_estimators': 10000,
    'lr': 0.01,
    'base_lr': 0.1,
    'focal_alpha': 0.5,
    'focal_gamma': 4
}
model = LightGBDT(**params)
model.fit(X_train, Y_train, 
    eval_set = [(X_test, Y_test)]
)
Y_pred = model.predict(X_test)
score = f1_score(Y_test, Y_pred)
print(f'F1-score = {score}')

# Save model
print('Saving model...')
joblib.dump(model, 'checkpoint/gbdt-model.pkl')
joblib.dump(encoder, 'checkpoint/encoder.pkl')

# Send email
print('Sending gmail...')
msg = MIMEMultipart()
msg['From'] = username
msg['To'] = username
msg['Subject'] = 'LightGBM訓練完成!!'
body = f'{str(params)}\n\nF1-score={score}'
msg.attach(MIMEText(body, 'plain'))
text = msg.as_string()
server = smtplib.SMTP('smtp.gmail.com', 587)
server.starttls()
server.login(username, password)
server.sendmail(msg['From'], msg['To'], text)
server.quit()
