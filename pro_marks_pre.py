import pandas as pd 
# import numpy as np 
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,root_mean_squared_error
import streamlit as st
import matplotlib.pyplot as plt
final_student=pd.read_csv("student_final_marks.csv")
x=final_student[["Attendance (%)","Internal Test 1 (out of 40)","Internal Test 2 (out of 40)","Assignment Score (out of 10)","Daily Study Hours"]]
y=final_student["Final Exam Marks (out of 100)"]
x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8,random_state=42)
# print(f"your x_train dataset is :\n{x_train.head()}")
standar_Scale=StandardScaler()
x_train_scaled=standar_Scale.fit_transform(x_train)
x_test_scaled=standar_Scale.fit_transform(x_test)
model=LinearRegression()
model_train=model.fit(x_train_scaled,y_train)
st.image(r"c:\Users\ASUS\OneDrive\Desktop\iit\new ai project top image .png")
st.sidebar.image(r"c:\Users\ASUS\OneDrive\Desktop\iit\side_image_3.jpeg")
st.sidebar.image(r"c:\Users\ASUS\OneDrive\Desktop\iit\Gemini_Generated_Image_1gw2xa1gw2xa1gw2 new 2.png")
st.sidebar.image(r"c:\Users\ASUS\OneDrive\Desktop\iit\side_new_1.jpeg")
st.header("Marks Prediction App",text_alignment="center",)
st.subheader("Enter Your Details below 👇",text_alignment="center")
# try:
co_1,col_2=st.columns(2)
with co_1:
    attendance=st.number_input("Enter your Attendance (%):\n",min_value=0,max_value=100)
    internal_test_1st=st.number_input("Enter your 1st Internal_test_score (Marks out of 40) :\n",min_value=0,max_value=40)
    internal_test_2nd=st.number_input("Ender your 2nd Internal_test_score (Marks out of 40) :\n",min_value=0,max_value=40)
    assignment_score=st.number_input("Enter your Assignment_score (out of 10) :\n",min_value=0,max_value=10)
    study_hours=st.number_input("Enter your Daily_Study_hours :\n",min_value=0.0)
# except:
    # print("Enter valid input (All input should be in intger / numbers form )")
user_information=pd.DataFrame([[attendance,internal_test_1st,internal_test_2nd,assignment_score,study_hours]],columns=["Attendance (%)","Internal Test 1 (out of 40)","Internal Test 2 (out of 40)","Assignment Score (out of 10)","Daily Study Hours"])
# st=StandardScaler()
user_scaled_info=standar_Scale.transform(user_information)

predict_y=model_train.predict(user_scaled_info)


if st.button("Predict now",type="primary"):
 predict_y=model_train.predict(user_scaled_info)[0]
if predict_y >=100: 
    st.success("Congrats your scored full makrs 100 out of 100🥳🏅🎉🎉 ")
elif predict_y <100 and predict_y >=60:
     st.success(f"Congrats you will passed with 1st devion in your final exam and your marks will be :{predict_y:.2f} 😊😊")
elif predict_y <=60 and predict_y >=33:
    st.success(f"😊 You will passed with 2nd devion in your final exam and your marks will be :\n{predict_y:.2f}")
else:
    st.success(f"😒 Ooooo sorry you not able to clear/pass your final exam and your marks will be :\n {predict_y}")
