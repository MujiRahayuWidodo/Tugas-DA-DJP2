import streamlit as st 
import pandas as pd
    
def main() : 
    # Membuat tab bar dengan dua tab
    tabs = st.tabs(["Data", "Profile"])

    with tabs[0]:
        data()
    with tabs[1]:
        profile()
        
if __name__ == '__main__' : 
  main()
