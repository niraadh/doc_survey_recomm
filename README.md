# Doctor Availability Email Notification System

## Overview
This project analyzes a dataset of doctors' availability and automatically sends email notifications to all doctors who are active at a given user-specified time.  
It can be used by hospitals, clinics, or healthcare platforms to notify available doctors about upcoming surveys, appointments, or important updates.

---

## Features
- Load and analyze a dataset containing doctors' online/offline timings.
- Filter doctors who are active at a specific time.
- Send automated emails to all active doctors.
- User-friendly web interface built with **Streamlit**.

---

## Tech Stack
- **Python 3.11+**
- **Streamlit** – for web interface  
- **Pandas** – for dataset analysis  
- **smtplib** / **email.mime** – for sending emails  
- **dotenv** – for handling sensitive email credentials  

---

## Project Structure
├── data/
   └── doctors_schedule.csv # Dataset of doctors and their online timings
├── app.py # Main Streamlit app
├── email_service.py # Email sending logic
├── requirements.txt # Required Python libraries
└── README.md # Project documentation


---

## Dataset Format
The dataset (`doctors_schedule.csv`) should have the following columns:

| Doctor_ID | Name         | Start_Time | End_Time |
|-----------|--------------|-----------|----------|
| D001      | Dr. Sharma   | 09:00     | 13:00    |
| D002      | Dr. Patel    | 14:00     | 18:00    |

- **Start_Time** and **End_Time** are in `HH:MM` 24-hour format.
- All times are assumed to be in the same timezone.

---

