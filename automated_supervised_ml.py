import streamlit as st 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import pycaret.classification as pc
import pycaret.regression as pr



def visual(df,keys,graph_keys):
        tab1, tab2, tab3, tab4, tab5 = st.tabs(['Scatter Plot', 'Histogram Plot', 'Box Plot', 'Bar Plot', "Pie plot"]) 
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            with col1:
                x_column = st.selectbox('Select column on x axis: ',num_cols,key=keys[0])
            with col2:
                y_column = st.selectbox('Select column on y axis: ',num_cols,key=keys[1])
            with col3:
                size = st.selectbox('Select column for size: ', num_cols,key=keys[2])
            with col4:
                color = st.selectbox('Select column for color: ',df.columns,key=keys[3])
            fig_scatter = px.scatter(df, x=x_column, y=y_column,color=color, size=size,)
            st.plotly_chart(fig_scatter,key=graph_keys[0])
        with tab2:
            x_hist = st.selectbox('Select feature to drow histogram plot: ',df.columns,key=keys[4])
            fig_hist=px.histogram(df,x=x_hist)
            st.plotly_chart(fig_hist,key=graph_keys[1])        
        with tab3:
            x_box = st.selectbox('Select feature to draw box plot: ', df.columns,key=keys[5])
            fig_box = px.box(df, y=x_box)
            st.plotly_chart(fig_box,key=graph_keys[2])
        with tab4:      
            col1, col2 = st.columns(2)
            with col1:
                x_column = st.selectbox('Select column on x axis: ',df.columns,key=keys[6])
            with col2:
                color = st.selectbox('Select column for color: ',df.columns,key=keys[7])
            fig_bar = px.bar(df, x=x_column, color=color)
            st.plotly_chart(fig_bar,key=graph_keys[3])
        with tab5:
            x_bar = st.selectbox('Select feature to draw pie plot: ', df.columns.tolist(),key=keys[8])
            fig_pie =px.pie(df,x_bar,).update_traces(textposition='inside', textinfo='percent').update_layout(uniformtext_minsize=10, uniformtext_mode='hide')
            st.plotly_chart(fig_pie,key=graph_keys[4])


st.markdown("<h1 style='text-align: center;'>Automated Supervised ML</h1>", unsafe_allow_html=True)
link='''https://github.com/AhmedMOM3/automated-supervised-ml'''

with st.expander('About this app'):
    st.write('This app helps you to perform EDA on your data, show visualization then train differient model and comparing between them using training report supported with some model visualization.')
    st.write(f'#<a target="_blank" href="{link}">`github link`</a>', unsafe_allow_html=True)


x=False

#uploading and reading data
data = st.sidebar.file_uploader('Upload a CSV or excel file')

#checkin if the file is valied or not
if data:
    st.sidebar.success('uploaded successfully')
    if data.name.split('.')[-1] == 'csv':
        df = pd.read_csv(data)
        x=True
    elif data.name.split('.')[-1] == 'xlsx':
        df = pd.read_excel(data)
        x=True
    else:
        st.error('please enter a valied file')
        x=False
else:
    st.sidebar.info('☝️ Upload a CSV file')

if x==True:
    if 'df' not in st.session_state:
        st.session_state.df = df
    # Initialize session state variables for model persistence
    if 'has_trained_model' not in st.session_state:
        st.session_state.has_trained_model = False
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'model_results' not in st.session_state:
        st.session_state.model_results = None
    if 'model_type' not in st.session_state:
        st.session_state.model_type = None
    
    upload_new_data=st.sidebar.button("Press here if you uploaded a new data",key='new data')
    if upload_new_data:
        st.session_state.df=df
        st.session_state.has_trained_model = False
        st.session_state.best_model = None
        st.session_state.model_results = None
        st.session_state.model_type = None
    
    

    st.sidebar.write('---')

    st.session_state.df.drop_duplicates(inplace=True)
    container_original_data= st.container(border=True,key="container_original_data")            
    with container_original_data:
        tab1, tab2, tab3, tab4= st.tabs(["Data", "Unique values",'Description',"Null Values"])
        tab1.subheader("The Data")
        tab1.write(df)
        tab2.subheader("The Number Of Unique Values For Each Column")
        tab2.write(pd.DataFrame(df.nunique()).T)
        tab3.subheader("Description Of Data")
        tab3.write(pd.DataFrame(df.describe()))
        tab4.subheader("Null Values For Data")
        tab4.write(pd.DataFrame(df.isnull().sum()))    
    
    st.sidebar.write("## Data preprocessing")
#reset button
    reset_button=st.sidebar.button("Reset All Processing Steps",key='reset')
    if reset_button:
        st.session_state.df = df
        st.sidebar.success("Reseted Successfully")
# drop unwanted columns
    st.sidebar.write("### •Droping unwanted columns")
    drop_or_not=st.sidebar.radio('Do you want to drop columns?', ['NO', 'YES'])
    if drop_or_not=="YES":
        options = st.sidebar.multiselect(
        'Pick columns to drop',
        st.session_state.df.columns,
        )
        drop_button=st.sidebar.button("drop")
        if drop_button:
            try:
                st.session_state.df.drop(options,axis=1,inplace=True)
                st.sidebar.success("Droped successfuly")
            except:
                st.sidebar.error("Drop error")

    st.sidebar.write('---')
    
#how to handel missing values
    st.sidebar.write("### •Handling missing values")
    left, right = st.columns(2)
    
    numerical_cols = st.session_state.df.select_dtypes(include=['int64', 'float64']).columns
    string_cols = st.session_state.df.select_dtypes(include=['object']).columns

    mean_columns = st.sidebar.multiselect(
        'Pick columns for `mean imputer`',
        numerical_cols,
        )
    median_columns = st.sidebar.multiselect(
        'Pick columns for `median imputer`',
        numerical_cols,
        )
    mode_columns = st.sidebar.multiselect(
        'Pick columns for `mode imputer`',
        st.session_state.df.columns,
        )
    
    impute_button=st.sidebar.button("Impute")

    mean_imputer  =SimpleImputer(strategy='mean', missing_values=np.nan)
    mode_imputer  =SimpleImputer(strategy='most_frequent', missing_values=np.nan)
    median_imputer=SimpleImputer(strategy='median', missing_values=np.nan)
    if impute_button:
        for item in mean_columns:
            try:
                st.session_state.df[item] = mean_imputer.fit_transform(st.session_state.df[item].values.reshape(-1,1))
            except:
                st.sidebar.error("mean error")
        for item in median_columns:
            try:
                st.session_state.df[item] = median_imputer.fit_transform(st.session_state.df[item].values.reshape(-1,1))
            except:
                st.sidebar.error("meadian error")
        for item in mode_columns:
            try:
                if (st.session_state.df[item].dtype=="O"):
                    st.session_state.df[item] = mode_imputer.fit_transform(st.session_state.df[item].values.reshape(-1,1))[:,0]
                else:
                    st.session_state.df[item] = mode_imputer.fit_transform(st.session_state.df[item].values.reshape(-1,1))
            except:
                st.sidebar.error("mode error")
        st.sidebar.success("Imputation completed successfully!")

    st.sidebar.write('---')
    
#data encoding
    st.sidebar.write("### •Categorical data Encoding")
    onehot_or_label=st.sidebar.radio('Do you want to perform:', ['One Hot Encoding', 'Label Encoding', 'Both One hot and Label Encoding'])

    string_cols = st.session_state.df.select_dtypes(include=['object']).columns
    Both=[]
    if onehot_or_label=="Both One hot and Label Encoding":
        Both = st.sidebar.multiselect(
                'Pick columns for `Label Encoding` \nthe other columns will be for `One Hot encoding` by default ',
                string_cols,
                )
    encode_button=st.sidebar.button("Encode")
    if encode_button:
        if onehot_or_label=='Label Encoding':
            for item in string_cols:
                le=LabelEncoder()
                try:
                    st.session_state.df[item] = le.fit_transform(st.session_state.df[item])
                    st.sidebar.success("Encoding completed successfully!")
                except:
                    st.sidebar.error("Label Encoder error")
        elif onehot_or_label=='One Hot Encoding':
            st.session_state.df=pd.get_dummies(st.session_state.df,drop_first=True)
            st.sidebar.success("Encoding completed successfully!")
        elif onehot_or_label=="Both One hot and Label Encoding":
            le=LabelEncoder()
            for item in Both:
                try:
                    st.session_state.df[item] = le.fit_transform(st.session_state.df[item])
                    st.sidebar.success("Encoding completed successfully!")
                except:
                    st.sidebar.error("Label Encoder error")
            
            st.session_state.df=pd.get_dummies(st.session_state.df,drop_first=True)

    st.sidebar.write("---")
    

#date after preprocessing
    container_data_Preprocessing= st.container(border=True,key="container_data_Preprocessing")            
    with container_data_Preprocessing:  
        st.write("## The Data After Preprocessing")
        st.dataframe(st.session_state.df)
                                       
#visualisation
    container_visual = st.container(border=True,key="container_visual")            
    with container_visual:  
        st.write("## Data Visualisation")
        tab1, tab2= st.tabs(['Original Data', 'After preprocessing']) 
        with tab1:
            keys=["b0","b1","b2","b3","b4","b5","b6","b7","b8"]
            graph_keys=["bg0","bg1","bg2","bg3","bg4"]
            visual(df,keys,graph_keys)
        with tab2:
            keys=["a0","a1","a2","a3","a4","a5","a6","a7","a8"]
            graph_keys=["ag0","ag1","ag2","ag3","ag4"]
            visual(st.session_state.df,keys,graph_keys)
            
    
#machine learning training
    st.sidebar.write("### •Machine Learning model training")
    target=st.sidebar.selectbox("Select Target",st.session_state.df.columns)
    select_button=st.sidebar.button("train model")
    

    
    if select_button:
        numerical_cols = st.session_state.df.select_dtypes(include=np.number).columns
        string_cols = st.session_state.df.select_dtypes(include=['object','bool']).columns
                
        if target in string_cols:
            st.session_state.model_type = 'classification'
            st.sidebar.info("Classification")

            with st.spinner("⏳ Please wait... Training The Model"):
                st.toast("⏳ Please wait... Training The Model")
                data = pc.setup(st.session_state.df, target=target, memory=False, transformation=True)
                st.session_state.best_model = pc.compare_models()
                st.session_state.model_results = pc.pull()
                st.session_state.has_trained_model = True
                st.sidebar.success("✅ Training complete!")
            
        elif target in numerical_cols:
            num_of_unique_values = len(st.session_state.df[target].unique())
            df_len = st.session_state.df.shape[0]
            
            if (num_of_unique_values < (df_len/10)):
                st.session_state.model_type = 'classification'
                st.sidebar.info("Classification")

                with st.spinner("⏳ Please wait... Training The Model"):
                    st.toast("⏳ Please wait... Training The Model")
                    data = pc.setup(st.session_state.df, target=target, memory=False)
                    st.session_state.best_model = pc.compare_models()
                    st.session_state.model_results = pc.pull()
                    st.session_state.has_trained_model = True
                    st.sidebar.success("✅ Training complete!")
            else:
                st.session_state.model_type = 'regression'
                st.sidebar.info("Regression")                
                
                with st.spinner("⏳ Please wait... Training The Model"):
                    st.toast("⏳ Please wait... Training The Model")
                    data = pr.setup(st.session_state.df, target=target, memory=False)
                    st.session_state.best_model = pr.compare_models()
                    st.session_state.model_results = pr.pull()
                    st.session_state.has_trained_model = True
                    st.sidebar.success("✅ Training complete!")
        else:
            st.sidebar.error("Not supported choose another column")


    # Display trained model results if available
    if st.session_state.has_trained_model:
        container_training = st.container(border=True, key="container_training")            
        with container_training:
            st.write("## Training report")
            st.dataframe(st.session_state.model_results)
            st.write('---')
            st.write(f"## Best model is: `{st.session_state.best_model.__class__.__name__}`")
            st.write('---')
            st.write(f"## Model Visualization")
            
            plot_dict_c={'Area Under the Curve':'auc','Discrimination Threshold':'threshold','Precision Recall Curve':"pr",'Class Prediction Error':'error','Classification Report':'class_report','Decision Boundary':'boundary','Learning Curve':'learning','Manifold Learning':'manifold','Calibration Curve':'calibration','Validation Curve':'vc','Dimension Learning':'dimension'}
            plot_dict_r={'Residuals Plot':'residuals','Prediction Error Plot':'error','Cooks Distance Plot':'cooks','Recursive Feat. Selection':'rfe','Learning Curve':'learning','Validation Curve':'vc','Manifold Learning':'manifold','Decision Tree':'tree'} 
            # Use a unique key for this selectbox
            if (st.session_state.model_type == 'classification'):
                plot_type = st.selectbox("Select plot type:", list(plot_dict_c.keys()), key="plot_type_selector")
            elif (st.session_state.model_type == 'regression'):
                plot_type = st.selectbox("Select plot type:", list(plot_dict_r.keys()), key="plot_type_selector")
            
            # Generate the plot based on model type and selected visualization
            try:
                if (st.session_state.model_type == 'classification'):
                    pc.plot_model(st.session_state.best_model, plot=plot_dict_c[plot_type], display_format="streamlit")
                elif (st.session_state.model_type == 'regression'):
                    pr.plot_model(st.session_state.best_model, plot=plot_dict_r[plot_type], display_format="streamlit")
            except Exception as e:
                st.error(f"Error generating plot: {str(e)}")
                st.info("This plot type might not be available for the current model or dataset. Please try another visualization.")
