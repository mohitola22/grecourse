# Core Pkg
import streamlit as st 
import streamlit.components.v1 as stc 
import pandas as pd
import time  
from PIL import Image
timestr=time.strftime("%Y/%m/%d-%H:%M:%S")
#import sqlite3
#conn=sqlite3.connect('data.db')
#c=conn.cursor()

#def create_usertable():
#    c.execute('create table if not exists userstable(username TEXT,password TEXT)')
    
#def add_userdata(username,password):
#    c.execute('insert into userstable(username,password) values(?,?)',(username,password))
#    conn.commit()

#def login_user(username,password):
#    c.execute('select * from userstable where username=? and password=?',(username,password))
#    data=c.fetchall()
#    return data
#def view_all_users():
#    c.execute('select username,password  from userstable')
#    data=c.fetchall()
#    return data
# Load EDA
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity,linear_kernel

import base64
# Load Our Dataset
def load_data(data):
	df = pd.read_csv(data)
	return df 


# Fxn
# Vectorize + Cosine Similarity Matrix

# =============================================================================
# count_vect = CountVectorizer()
# print(count_vect)
# df = load_data("C:/Users/mohit.verma/Desktop/New folder (3)/data/udemy_course_data.csv")
# data=df['course_title']
# cv_mat = count_vect.fit_transform(data)
# print(cv_mat)
# # Get the cosine
# cosine_sim_mat = cosine_similarity(cv_mat)
# print(cosine_sim_mat)
# =============================================================================


def vectorize_text_to_cosine_mat(data):
	count_vect = CountVectorizer()
	cv_mat = count_vect.fit_transform(data)
	# Get the cosine
	cosine_sim_mat = cosine_similarity(cv_mat)
	return cosine_sim_mat

# =============================================================================
# df = load_data("C:/Users/mohit.verma/Desktop/New folder (3)/data/udemy_course_data.csv")
# course_indices = pd.Series(df.index,index=df['course_title']).drop_duplicates()
# print(course_indices)
# =============================================================================
# Recommendation Sys

@st.cache
def get_recommendation(title,cosine_sim_mat,df,num_of_rec=10):
	# indices of the course
	course_indices = df['Title'].tolist()
	# Index of course
	L = course_indices.index(title)
    # Look into the cosine matr for that index
	sim_scores =list(enumerate(cosine_sim_mat[L]))
	sim_scores = sorted(sim_scores,key=lambda x: x[1],reverse=True)
	selected_course_indices = [i[0] for i in sim_scores[1:]]
	selected_course_scores = [i[1] for i in sim_scores[1:]]

	# Get the dataframe & title
	result_df = df.iloc[selected_course_indices]
	result_df['similarity_score'] = selected_course_scores
	final_recommended_courses = result_df[['Title','similarity_score','CourseLink','Price','TotalEnrolled','Rating','ReviewedBy']]
	return final_recommended_courses.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>
<p style="color:blue;"><span style="color:black;">👍 Rating:</span>{},<span style="color:black;"> 🖋 ReviewedBy: </span>{}</p>

</div>
"""

# Search For Course 
@st.cache
def search_term_if_not_found(term,df):
	result_df = df[df['Title'].str.contains(term,case=False)]
	return result_df

def get_table_download_link(df2):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """ 
    csv = df2.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
    new_filename="datafile_{}.csv".format(timestr)
    href = f'<a href="data:file/csv;base64,{b64}" download="{new_filename}">Download csv file</a>'
    return href
    
def main():
    #img = Image.open("student.jpeg")
    #st.image(img,width=None)
    #img2 = Image.open("images.jpg")
    

    #st.image(img2,width=None)
    menu = ["Course Recommendation","College Predictor","About"]
    choice = st.sidebar.selectbox("Menu",menu)
    df = load_data("udemy.csv")
    df.drop_duplicates(subset ="Title",keep= 'first', inplace = True)
    #print(pd.unique(df['Title']))
    #print(list(df.iloc[0]))
    df['Title']=df['Title'].str.strip()+" "+df['ProvidedBy']
    
    
    data=df['Title']
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    print(cv_mat)
    
    
    #print(df['Title'].head(10))
    N_R=((df['Rating']*df['ReviewedBy']).sum())/df['ReviewedBy'].sum()
    st.write('Avg rating is ',N_R)
    #st.write('d')
    hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
    
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
position: absolute;
left: 0;
bottom: 0;
width: 100%;
background-color: white;
color: black;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤ by <a style='display: block; text-align: center;' href="https://one.motherson.com/SitePages/main.aspx#monepage[1,1]" target="_blank">Motherson</a></p>
</div>
"""
    
    
    df[['Price']]=df[['Price']].apply(pd.to_numeric)
    if(choice=='College Predictor'):
        st.title("College Predictor App")
        with st.form(key='form1'):
            rank=st.text_input("Enter your rank",value="10")
            caste=st.selectbox("Caste",["General","OBC","SC","ST"])
            college=list(st.multiselect("College",['NIT Calcut', 'NIT Surathkal', 'NIT Tiruchirappalli','VNIT Nagpur', 'SVNIT Surat', 'MNNIT Allahabad', 'NIT Delhi','MNIT Jaipur', 'NIT Manipur', 'NIT Durgapur', 'NIT Warangal','MANIT Bhopal', 'NIT Patna','All'],default="All"))
           
            gender=st.selectbox("Gender", ['Male','Female'])
            state=st.selectbox("State",['Other State','Home State'])
            course=list(st.multiselect("Branch",['Computer Science and Engineering','Electronics and Communication Engineering','Mechanical Engineering','Electrical and Electronics Engineering', 'Civil Engineering','Electrical Engineering']))
            sumbit_button=st.form_submit_button(label='Çollege Predictor')
        if(sumbit_button):
            st.success("Hello you {}".format(state))
               
        df1=load_data("C:/Users/mohit.verma/Desktop/New folder (4)/data.csv")
        df2=df1[(df1.Category==caste) & ((df1['College'].isin(college)) | (college[0]=='All')) & (df1.Closing_rank>=int(rank)) & (df1['Course_Name'].isin(course)) & (df1.Gender==gender) & (df1.State==state) ].sort_values(by=['Closing_rank'],ascending=False)
        st.dataframe(df2)
        st.markdown(get_table_download_link(df2), unsafe_allow_html=True)
        st.markdown('<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMzM2M2IyN2QtMWU5Ny00OWRlLTlhM2QtNTFmN2UwOTg1OWU2IiwidCI6IjdhNzQ2NzQyLTc5MzEtNGY1ZC04YWYzLWQyYmY2ZmJiM2ExNSIsImMiOjh9" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)
    
   
    elif choice == "Course Recommendation":
        st.title("Course Recommendation App")
        st.subheader("Recommend Courses")
        search_term = st.text_input("Search")
        Level=list(st.multiselect("Difficulty",['all','Beginner Level','Intermediate Level','Expert Level','All levels'],default="all"))

        Rating=st.number_input('Please enter min rating',min_value=1.0,max_value=10.0,step=0.1)
        a=df['Price'].max()
        b=df['ReviewedBy'].max()
        c=df['TotalEnrolled'].max()
        
        
        price=st.slider("Max Price Allowed:",min_value=0,max_value=a,step=10)
        #st.write(price)
        ReviewedBy=st.slider("Min ReviewedBy:",min_value=0,max_value=b,step=10)
        TotalEnrolled=st.slider("Min TotalEnrolled",min_value=0,max_value=c,step=10)
# =============================================================================
#         from streamlit_toggle import st_toggleswitch
#         toggle=st_toggleswitch('Paid',key=1)
#         if toggle:
#             st.write('haha')
# =============================================================================
        #private onChange = (e: React.ChangeEvent<HTMLInputElement>): void => {const isChecked = e.currentTarget.checked this.setState({ value: isChecked }, () => { Streamlit.setComponentValue(isChecked)})
        #df=df.where(df['level'] in Level)
        #df=df[(df['level']=="Beginner Level")]
# =============================================================================
#         df[['Rating']]=df[['Rating']].apply(pd.to_numeric)
#         df[['Price']]=df[['Price']].apply(pd.to_numeric)
#         df=df[(df['Difficulty'].isin(Level)  | ('all' in Level) ) & (df['Rating']>=Rating) & (df['Price']<=price)]
# =============================================================================
        
        x=df.shape[0]
        #print(x)
        #print(list(df[0][:]))        
        df.loc[x]=[100, 'GRE 46 Hours Math Prep Target GRE 330 ', 'Jackson K', 4.6, '664', 'All Levels', 1, 3320, 'https://www.udemy.com/course/master-gre-math/', 0]
        df[['Price']]=df[['Price']].apply(pd.to_numeric)
        df[['Rating']]=df[['Rating']].apply(pd.to_numeric)
        df[['ReviewedBy']]=df[['ReviewedBy']].apply(pd.to_numeric)
        df[['TotalEnrolled']]=df[['TotalEnrolled']].apply(pd.to_numeric)

        #print(df[['Price']])
        df=df[(df['Difficulty'].isin(Level)  | ('all' in Level) ) & ((df['Rating']>=Rating)) & (df['Price']<=price) & (df['ReviewedBy']>=ReviewedBy) & (df['TotalEnrolled']>=TotalEnrolled)] #ERROR COLUMN ROW DIFF KI VAJAH SE SAYAD ERROR AA RHI H.
        #df.loc[x] = [1113822.0, 'Complete GST Course & Certification - Grow Your CA Practice', 'https://www.udemy.com/goods-and-services-tax/', True, '75', 2792.0, 923.0, 274.0, 'All Levels', '39 hours', '2017-03-09T16:34:20Z', 'Business Finance', 209400.0, '09-03-2017', '16:34:20Z', 2017.0, 3.0, 9.0]
        df['Title'][x]='ABC'
        df['Title'][x]=search_term 
        cosine_sim_mat = vectorize_text_to_cosine_mat(df['Title'])
        #st.dataframe(df)
        num_of_rec = st.sidebar.number_input("Number",4,30,7)
        if st.button("Recommend"):
             if search_term is not None:
                 try:
                    results = get_recommendation(search_term,cosine_sim_mat,df,num_of_rec)
                    results[['TotalEnrolled']]=results[['TotalEnrolled']].apply(pd.to_numeric)
                    with st.beta_expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]
                        rating=row[1][5]
                        reviewby=row[1][6]

						# st.write("Title",rec_title,)
                        stc.html(RESULT_TEMP.format(rec_title,rec_score,rec_url,rec_price,rec_num_sub,rating,reviewby),height=350)
                    df.drop ([x], axis=0, inplace = True)
                    st.markdown(get_table_download_link(results), unsafe_allow_html=True)
                    #st.markdown('<iframe width="600" height="373.5" src="https://app.powerbi.com/view?r=eyJrIjoiMzM2M2IyN2QtMWU5Ny00OWRlLTlhM2QtNTFmN2UwOTg1OWU2IiwidCI6IjdhNzQ2NzQyLTc5MzEtNGY1ZC04YWYzLWQyYmY2ZmJiM2ExNSIsImMiOjh9" frameborder="0" allowFullScreen="true"></iframe>',unsafe_allow_html=True)
                    #st.image(img2,width=None)
                    st.markdown(footer,unsafe_allow_html=True)

                 except:
                    results= "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term,df)
                    st.dataframe(result_df)



				# How To Maximize Your Profits Options Trading




    else:
        st.subheader("About")
        st.text("Built with Streamlit & Pandas")


if __name__ == '__main__':
	main()


