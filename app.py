# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 20:47:07 2021

@author: Sathiya vigraman M
"""

import streamlit as st
import pickle
import numpy as np
import  pandas as pd
import matplotlib.pyplot as plt#library for plots
import requests
from bs4 import BeautifulSoup as bs
from dateutil.relativedelta import *



module = pickle.load(open('SGB.pkl', 'rb'))
module_sbi = pickle.load(open('SBI.pkl', 'rb'))
module_irfc = pickle.load(open('IRFC.pkl', 'rb'))



#Scrapping SGB Data
page = requests.get('https://www.topsharebrokers.com/report/sovereign-gold-bond-return-calculator/317/')
soup = bs(page.content, 'html.parser')
tbl = soup.find("table")
data_frame = pd.read_html(str(tbl))[0]
buy_price = data_frame.iloc[0, 3]
buy_date = data_frame.iloc[0, 2]
buy_date = pd.to_datetime(buy_date)



def interest_SGB(half_year_average, redemption_value):

    #interest
    total_interest = 0
    for i in range(0, len(half_year_average)):
        j = 0
        j = half_year_average[i] * ((2.5/2)/100)
        total_interest += j 
    return_value = redemption_value + total_interest
    profit_percentage_online = ((return_value - (buy_price))/(buy_price)) * 100
    return profit_percentage_online




def interest_SBI(buy_value, redemption_value, year_average):

    #interest
    total_interest = 0
    for i in range(0, len(year_average)):
        j = 0
        j = year_average[i] * (4.6/100)
        total_interest += j 
    return_value = redemption_value + total_interest
    profit_percentage_online = ((return_value - (buy_value))/(buy_value)) * 100
    return profit_percentage_online




def interest_IRFC(buy_value, redemption_value, year_average):

    #interest
    total_interest = 0
    for i in range(0, len(year_average)):
        j = 0
        j = year_average[i] * (8/100)
        total_interest += j 
    return_value = redemption_value + total_interest
    profit_percentage_online = ((return_value - (buy_value))/(buy_value)) * 100
    return profit_percentage_online




def main():
    #set_png_as_page_bg('background.png')
    st.set_page_config(page_title = 'innodadatics', page_icon = ":alien:")
    st.title("Investment on Bonds")
    html_temp = """
    <div style="background-color:#025246 ;padding:10px">
    <h2 style="color:white;text-align:center;">Comparing Profit of Sovereign Gold Bond with General Bonds (Indian Railway Finance Corporation & SBI Life (Unit II Regular) - Bond Fund)</h2>
    </div>
    """
    hide_menu_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_menu_style, unsafe_allow_html=True)
    st.markdown(html_temp, unsafe_allow_html=True)
    menu = ['Forecast', 'About Project', 'About us']
    choice = st.sidebar.selectbox("Menu", menu)
    if choice == 'Forecast':
        year = st.selectbox("how many years you plan for investment: ", np.arange(5, 9, 1))    
        if st.button("Forecast"):
            maturity_date = buy_date + relativedelta(years =+ year)
            fc_series = module.forecast(len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))))[0]
            fc_series = pd.Series(fc_series, index = (pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))), freq='D')))
            redemption_value = fc_series[fc_series.index.max()]
            half_year_average = fc_series.resample('6M').mean()
            value_1 =  round(interest_SGB(half_year_average, redemption_value), 2)
            fc_series_1 =  module_sbi.forecast(len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))))[0]
            fc_series_1 = pd.Series(fc_series_1, index = (pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))), freq='D')))
            redemption_value_1 = fc_series_1[fc_series_1.index.max()]
            buy_price_1 = fc_series_1[fc_series_1.index.min()]
            year_average_1 = fc_series_1.resample('12M').mean()
            value_3 = round(interest_SBI(buy_price_1, redemption_value_1, year_average_1), 2)
            fc_series_2 = module_irfc.forecast(len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))))[0]
            fc_series_2 = pd.Series(fc_series_2, index = (pd.date_range(start = pd.to_datetime('today').date(), periods = len(pd.DataFrame(pd.date_range(start = buy_date, end = maturity_date))), freq='D')))
            redemption_value_2 = fc_series_2[fc_series_2.index.max()]
            buy_price_2 = fc_series_2[fc_series_2.index.min()]
            year_average_2 = fc_series_1.resample('12M').mean()
            value_4 = round(interest_IRFC(buy_price_2, redemption_value_2, year_average_2), 2)
            st.success(f'The profit percentage for next {year} years in Sovereign Gold Bond is {value_1}%.'.format(year, value_1))
            st.success(f'The profit percentage for next {year} years in SBI Life (Unit II Regular) - Bond Fund Bond is {value_3}%'.format(year,    value_3))
            st.success(f'The profit percentage for next {year} years in Indian Railway Finance Corporation - N1 Bond is {value_4}%'.format(year,   value_4))
            compare = {'Sovereign Gold Bond': value_1, 'SBI Life (Unit II Regular) - Bond Fund': value_3, 'Indian Railway Finance Corporation - N1 Seriesx': value_4}
            result_1 = max(compare, key = compare.get)
            result_2 = max(value_1, value_3, value_4)
            st.success(f'Best investment in bond for next {year} years is {result_1} and profit percentage is {result_2}%'.format(year, result_1, result_2))
       
            plot1 = pd.concat([fc_series, fc_series_1, fc_series_2], axis=1)
            plot1.columns = ['Sovereign Gold Bond', 'SBI Life (Unit II Regular) - Bond Fund', 'Indian Railway Finance Corporation - N1 Series']
            st.line_chart(plot1)
            fig = plt.figure(figsize = (10, 5))
            # creating the bar plot
            plt.bar(list(compare.keys()), list(compare.values()), color ='red', width = 0.2)
            plt.xlabel("Bonds")
            plt.ylabel("Profit Percentage")
            plt.title("Investment in Bonds")
            plt.show()
            st.pyplot(fig)
    elif  choice == 'About Project':
            st.subheader(" Overview:")
            st.write('An investment is an asset or item acquired with the goal of generating income or appreciation. Appreciation refers to an increase in the value of an asset over time. "Investing" we mean buying an asset for making a profit by selling it in the future, after it appreciates in value. Here we are understanding the value of asset by invest in Bonds')
            st.write('1)Gold Bonds')
            st.write('2)General Bonds')
            st.subheader("Process:")
            st.write('Investment in bonds requires lot of research in terms of returns received over a period of tenure. Although forecasting statistics will help the user to see the estimated returns of Bonds purchased and also understand the returns on the amount invested in certain bond.')
            st.subheader(" Scope/Insights:")
            st.write('It helps the user to select the appropriate bond for investment and educate the user regarding investment risk along with maximum returns.')
            st.header(" GOAL:")
            st.write('Finding the best investment between Gold Bonds and General Bonds by building a forecasting model.')
            st.subheader("Objective:")
            st.write('Maximize the Investment returns')
            st.subheader("Constraint:")
            st.write('Minimize the Investment risk')
            st.subheader("Data Collection:")
            st.write('In Gold Bond, we have collected Standard (physical) gold bonds and Sovereign Gold Bond dataset whereas in General bonds, collected SBI Life (Unit II Regular) - Bond Fund and Indian Railway Finance Corporation dataset.')
            st.subheader("Forecasting Model:")
            st.write('We are here use ARIMA model for forecasting and streamlit tool for deployment.')

            
    elif choice == 'About us':
            st.header('Below mentioed peoples are working on this project')
            
            st.subheader("[Raju G] (http://www.linkedin.com/in/raju-gulla-728770217/) - Mentor")
            st.header('Members:')
            st.subheader("[Mugesh G] (http://www.linkedin.com/in/mugesh-g)")
            st.subheader("[Sathiyavigraman M] (https://www.linkedin.com/in/sathiyavigraman) ")
            st.subheader("[Anubhav] (https://www.linkedin.com/in/anubhav-32993) ")
            st.subheader("[Rakesh Reddy Kadapala] (https://www.linkedin.com/in/kadapalarakeshreddy)     ")
            st.subheader("[Neha Shirbhate] (https://www.linkedin.com/in/neha-shirbhate-0aa47762) ")
            st.subheader("[Ayesha Siddiqha] (https://www.linkedin.com/in/ayesha-siddiqha-52a4821aa) ")
            st.subheader("[Ayush Agrawal] (https://www.linkedin.com/in/ayush-agrawal-91038659) ")
            st.subheader("[Aakrit Pai] (https://www.linkedin.com/in/aakrit-pai-9702a9144) ")
            st.subheader("[Shubham Undirwade] (https://www.linkedin.com/in/shubham-undirwade-157782b4)")
            st.subheader("[Varun Kumar] (https://www.linkedin.com/in/varun-kumar-19b1a843) ")
            st.subheader("[Naushina Farheen S] (https://www.linkedin.com/in/naushina-shaik-91587b198) ")
            st.subheader("[Jaspal Singh] (https://www.linkedin.com/in/jaspal-singh-19a3a813) ")
            st.subheader("[Saikumar Godha] (https://www.linkedin.com/in/saikumar-godha-42081115a) ")
            st.subheader("[Yash Bohra] (http://linkedin.com/in/yash-bohra-014b15197) ")
            st.subheader("[Paritala Harshitha] (https://www.linkedin.com/in/harshitha-paritala1) ")
            st.subheader("[Abdulkadhar] (https://www.linkedin.com/in/abdulkadar-shaikh-46bbb3190) ")
            st.subheader("[B.DivyaSuma] (www.linkedin.com/in/divya-suma-697a0517b) ")
            st.subheader("[Ashish Singh] (https://www.linkedin.com/in/ashish-singh-684b10194) ")
            st.subheader("[Madhuri Mattarparthy] (https://www.linkedin.com/in/madhuri-mattaparthy-0a8562138)  ")
            st.subheader("[Mamta Thakur] (https://www.linkedin.com/in/mamta-thakur-35b36221a)")
            st.subheader("[Akshay Kumar] (https://www.linkedin.com/in/akshay-kumar-08)")

            
            



if __name__=='__main__':
    main()






