import streamlit as st
import time
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import multiprocessing as mp
import numpy as np
import pandas as pd
from data_utils import *
import csv
from config_fn import *

from os import path
st.set_page_config(page_title="Covid Help",page_icon="🧊",layout="wide")

def make_clickable(url, text):
    return f'<a target="_blank" href="{url}">{text}</a>'


hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)



def display_app_header(main_txt,sub_txt,is_sidebar = False):
    """
    function to display major headers at user interface
    Parameters
    ----------
    main_txt: str -> the major text to be displayed
    sub_txt: str -> the minor text to be displayed
    is_sidebar: bool -> check if its side panel or major panel
    """

    html_temp = f"""
    <div style = "background.color:#3c403f  ; padding:15px">
    <h2 style = "color:white; text_align:center;"> {main_txt} </h2>
    <p style = "color:white; text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html = True)
    else:
        st.markdown(html_temp, unsafe_allow_html = True)


display_app_header('Covid Resource Search Engine','This website is designed to help friends & relatives of Covid affected patients. You can enter your requirement and location, will try to fetch the most appropriate results.',is_sidebar = False)
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>

"""
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)
#st.markdown("<h1 style='text-align: center; color: red;'>Some title</h1>", unsafe_allow_html=True)
#st.header('Covid resources search:')
#st.subheader('This website is designed to help friends & relatives of Covid affected patients. You can enter your requirement and location, will try to fetch the most appropriate results.')

with st.beta_expander("Click here to know how it works ?"):
    st.write("This Solution is trying to focus on two major problems - 1. Bring the latest results 2. Most Appropriate Results without looking into too much information.\n The website is pulling the tweets on real time and identifying if its providing any details on Covid resources and extract the relevant details from it.\n When you enter a search/requirement it tries to understand your requirement & location to bring the most optimal results to you. ")

st.subheader('')

with st.form(key='my_form'):
	sentence = st.text_input(label='Input your search (For ex: Need remdesivir for my brother in mumbai):')
	submit_button = st.form_submit_button(label='Search')



#sentence = st.text_input('Input your search (For ex: Need remdesivir for my brother in mumbai):')

if sentence:
    time_sen = time.time()
    sen_list = [time_sen,sentence]
    print(sen_list)
    with open('recent_searches.csv', 'a',newline="") as f:
        write = csv.writer(f)
        write.writerow(sen_list)
    with st.spinner('Give us 10-15 seconds, we are fetching & analysing the results for you..'):
        data_to_view_total = t_main(sentence)
        st.success('Done!')

    #st.write(data_to_view)
    if data_to_view_total.empty:
        st.write('No result found. Give it one more try and please be more specific in your search! - Mention Requirement along with the city')
    else:
        data_to_view_total['validation_status'] = data_to_view_total['validation_status'].str.replace('1', 'Verified')
        data_to_view_total['link'] = data_to_view_total['link'].apply(make_clickable, args = ('Tweet Link',))
        #df['validation_status'] = df['validation_status'].str.replace('', 'Verified')
        validate_list = ['Verified','2','3']

        data_to_view_verified = data_to_view_total[data_to_view_total['validation_status'] == 'Verified']
        data_to_view_unverified = data_to_view_total[~data_to_view_total['validation_status'].isin(validate_list)]
        if data_to_view_verified.empty:
            pass
        else:
            st.write("Verified Leads by our volunteers:")
            data_to_view_verified = data_to_view_verified.sort_values(by=['validated','score'], ascending=False)
            data_to_view_verified.reset_index(inplace = True)
            data_to_view_verified.drop(['index'], axis = 1, inplace=True)
            st.write(data_to_view_verified.head(15).to_html(escape = False), unsafe_allow_html = True)


        if data_to_view_unverified.empty:
            pass
        else:
            st.write("Leads:")
            data_to_view_unverified = data_to_view_unverified.sort_values(by=['time','score'], ascending=False)
            data_to_view_unverified.reset_index(inplace = True)
            data_to_view_unverified.drop(['index','validation_status', 'validation_details','validated'], axis = 1, inplace=True)
            st.write(data_to_view_unverified.head(15).to_html(escape = False), unsafe_allow_html = True)
    #st.table(data_to_view.to_html(escape=False, index=False))
    #st.table(data_to_view.assign(hack='').set_index('hack').to_html(escape=False, index=False))


st.markdown("""
<style>
.big-font {
    font-size:14px !important;
}
</style>
""", unsafe_allow_html=True)




#
# dataframe = np.random.randn(10, 20)
# #st.dataframe(dataframe)
#
# dataframe = pd.DataFrame(
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
#
# #st.dataframe(dataframe.style.highlight_max(axis=0))
#
#
# dataframe = pd.DataFrame(#
#     np.random.randn(10, 20),
#     columns=('col %d' % i for i in range(20)))
# #st.table(dataframe)
#
#
if path.exists('recent_searches.csv'):
    data_to_display_sen = pd.read_csv('recent_searches.csv',names=['sen_time','sentence'])
    data_to_display_sen = data_to_display_sen.sort_values(by=['sen_time'], ascending=False)
    data_to_display_sen.drop_duplicates(subset=['sentence'], keep='first',inplace=True,ignore_index=True)
    st.markdown('<p class="big-font">Recent searches on the platform</p>', unsafe_allow_html=True)
    st.dataframe(data_to_display_sen['sentence'].head(15))


footer="""<style>
a:link , a:visited{
color: blue;
background-color: transparent;
text-decoration: underline;
}

a:hover,  a:active {
color: red;
background-color: transparent;
text-decoration: underline;
}

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
'''<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://www.linkedin.com/in/shivamagrawal/" target="_blank">Shivam Agrawal</a></p>'''
</div>
"""
#st.subheader('')
#st.markdown(footer,unsafe_allow_html=True)
