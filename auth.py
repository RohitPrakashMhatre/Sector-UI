import streamlit as st

USERS = {
    "rohitmhatre3232@gmail.com": "rohit457",
    "Kedarmhatre5gmail.com": "kedar"
}

def login_screen():
    st.title("Login to the dashboard")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username in USERS and USERS[username] == password:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login successful! Loading app...")
            st.rerun()  # Refresh app to load main UI
        else:
            st.error("Invalid username or password")
