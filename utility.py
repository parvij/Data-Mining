#https://www.kaggle.com/kabure/eda-feat-engineering-encode-conquer




# time cost
%%time





traintest = pd.concat([train, test])
dummies = pd.get_dummies(traintest, columns=traintest.columns, drop_first=True, sparse=True)
train_ohe = dummies.iloc[:train.shape[0], :]
test_ohe = dummies.iloc[train.shape[0]:, :]
# It looks like `sparse = True` in `get_dummies` no longer makes anything sparse, and we have to explicitly convert
# If you don't do this, the model takes forever... it is much much faster on sparse data!
train_ohe = train_ohe.sparse.to_coo().tocsr()
test_ohe = test_ohe.sparse.to_coo().tocsr()





def resumetable(df):
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes,columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name','dtypes']]
    summary['Missing'] = df.isnull().sum().values    
    summary['Uniques'] = df.nunique().values
    summary['First Value'] = df.loc[0].values
    summary['Second Value'] = df.loc[1].values
    summary['Third Value'] = df.loc[2].values

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2) 

    return summary
    
    
    ## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
    
    
    
    
# visualization binary

def vis_bin(df,cols):
    #Looking the V's features
    import matplotlib.gridspec as gridspec # to do the grid of plots
    grid = gridspec.GridSpec(len(cols)/2, 2) # The grid of chart
    plt.figure(figsize=(16,20)) # size of figure

    # loop to get column and the count of plots
    for n, col in enumerate(df[cols]): 
        ax = plt.subplot(grid[n]) # feeding the figure of grid
        sns.countplot(x=col, data=df, hue='target', palette='hls') 
        ax.set_ylabel('Count', fontsize=15) # y axis label
        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label
        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label
        sizes=[] # Get highest values in y
        for p in ax.patches: # loop to all objects
            height = p.get_height()
            sizes.append(height)
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(height/total*100),
                    ha="center", fontsize=14) 
        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights

    plt.show()


    
    
    
    
    
#visualization for category cols and binary target
def ploting_cat_fet(df, cols, vis_row=5, vis_col=2,target='target'):
    
    grid = gridspec.GridSpec(vis_row,vis_col) # The grid of chart
    plt.figure(figsize=(17, 35)) # size of figure

    # loop to get column and the count of plots
    for n, col in enumerate(df_train[cols]): 
        tmp = pd.crosstab(df_train[col], df_train[target], normalize='index') * 100
        tmp = tmp.reset_index()
        tmp.rename(columns={0:'No',1:'Yes'}, inplace=True)

        ax = plt.subplot(grid[n]) # feeding the figure of grid
        sns.countplot(x=col, data=df_train, order=list(tmp[col].values) , color='green') 
        ax.set_ylabel('Count', fontsize=15) # y axis label
        ax.set_title(f'{col} Distribution by Target', fontsize=18) # title label
        ax.set_xlabel(f'{col} values', fontsize=15) # x axis label

        # twinX - to build a second yaxis
        gt = ax.twinx()
        gt = sns.pointplot(x=col, y='Yes', data=tmp,
                           order=list(tmp[col].values),
                           color='black', legend=False)
        gt.set_ylim(tmp['Yes'].min()-5,tmp['Yes'].max()*1.1)
        gt.set_ylabel("Target %True(1)", fontsize=16)
        sizes=[] # Get highest values in y
        for p in ax.patches: # loop to all objects
            height = p.get_height()
            sizes.append(height)
            ax.text(p.get_x()+p.get_width()/2.,
                    height + 3,
                    '{:1.2f}%'.format(height/total*100),
                    ha="center", fontsize=14) 
        ax.set_ylim(0, max(sizes) * 1.15) # set y limit based on highest heights


    plt.subplots_adjust(hspace = 0.5, wspace=.3)
    plt.show()
    
    
    
    
#prepareing ordinal feature    
# Importing categorical options of pandas
from pandas.api.types import CategoricalDtype 

# seting the orders of our ordinal features
ord_1 = CategoricalDtype(categories=['Novice', 'Contributor','Expert', 
                                     'Master', 'Grandmaster'], ordered=True)
ord_2 = CategoricalDtype(categories=['Freezing', 'Cold', 'Warm', 'Hot',
                                     'Boiling Hot', 'Lava Hot'], ordered=True)

# Transforming ordinal Features
df_train.ord_1 = df_train.ord_1.astype(ord_1)
df_train.ord_2 = df_train.ord_2.astype(ord_2)
# test dataset
df_test.ord_1 = df_test.ord_1.astype(ord_1)
df_test.ord_2 = df_test.ord_2.astype(ord_2)


# Geting the codes of ordinal categoy's - train
df_train.ord_1 = df_train.ord_1.cat.codes
df_train.ord_2 = df_train.ord_2.cat.codes

# Geting the codes of ordinal categoy's - test
df_test.ord_1 = df_test.ord_1.cat.codes
df_test.ord_2 = df_test.ord_2.cat.codes



# Transfer the cyclical features into two dimensional sin-cos features
# https://www.kaggle.com/avanwyk/encoding-cyclical-features-for-deep-learning

def date_cyc_enc(df, col, max_vals):
    df[col + '_sin'] = np.sin(2 * np.pi * df[col]/max_vals)
    df[col + '_cos'] = np.cos(2 * np.pi * df[col]/max_vals)
    return df










from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from sklearn.linear_model import LogisticRegression

# Model
def run_cv_model(train, test, target, model_fn, params={}, eval_fn=None, label='model'):
    kf = KFold(n_splits=5)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    i = 1
    for dev_index, val_index in fold_splits:
        print('Started ' + label + ' fold ' + str(i) + '/5')
        dev_X, val_X = train[dev_index], train[val_index]
        dev_y, val_y = target[dev_index], target[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y = model_fn(dev_X, dev_y, val_X, val_y, test, params2)
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score))
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / 5.0
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores}
    return results


def runLR(train_X, train_y, test_X, test_y, test_X2, params):
    print('Train LR')
    model = LogisticRegression(**params)
    model.fit(train_X, train_y)
    print('Predict 1/2')
    pred_test_y = model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2


lr_params = {'solver': 'lbfgs', 'C': 0.1}
results = run_cv_model(train_ohe, test_ohe, target, runLR, lr_params, auc, 'lr')














#Map cat vals which are not in both sets to single values

for col in cols:
    train_vals = set(df_train[col].unique())
    test_vals = set(df_test[col].unique())
   
    xor_cat_vals=train_vals ^ test_vals
    if xor_cat_vals:
        df.loc[df[col].isin(xor_cat_vals), col]="xor"


        
        
        
        
        
        
        
        
        
        
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score as auc
from catboost import Pool, CatBoostClassifier
from category_encoders import TargetEncoder


# Model
def run_cv_model(categorical_indices, train, test, target, model_fn, params={}, eval_fn=None, label='model', n_folds=5):
    kf = KFold(n_splits=n_folds)
    fold_splits = kf.split(train, target)
    cv_scores = []
    pred_full_test = 0
    pred_train = np.zeros((train.shape[0]))
    feature_importances = pd.DataFrame()
    feature_importances['feature'] = test.columns
    i = 1
    for dev_index, val_index in fold_splits:
        print('-------------------------------------------')
        print('Started ' + label + ' fold ' + str(i) + f'/{n_folds}')
        dev_X, val_X = train.iloc[dev_index], train.iloc[val_index]
        dev_y, val_y = target.iloc[dev_index], target.iloc[val_index]
        params2 = params.copy()
        pred_val_y, pred_test_y, fi = model_fn(categorical_indices, dev_X, dev_y, val_X, val_y, test, params2)
        feature_importances[f'fold_{i}'] = fi
        pred_full_test = pred_full_test + pred_test_y
        pred_train[val_index] = pred_val_y
        if eval_fn is not None:
            cv_score = eval_fn(val_y, pred_val_y)
            cv_scores.append(cv_score)
            print(label + ' cv score {}: {}'.format(i, cv_score), '\n')
        i += 1
    print('{} cv scores : {}'.format(label, cv_scores))
    print('{} cv mean score : {}'.format(label, np.mean(cv_scores)))
    print('{} cv std score : {}'.format(label, np.std(cv_scores)))
    pred_full_test = pred_full_test / n_folds
    results = {'label': label,
              'train': pred_train, 'test': pred_full_test,
              'cv': cv_scores, 'fi': feature_importances}
    return results


def runCAT(categorical_indices, train_X, train_y, test_X, test_y, test_X2, params):
    # Pool the data and specify the categorical feature indices
    print('Pool Data')
    _train = Pool(train_X, label=train_y, cat_features = categorical_indices)
    _valid = Pool(test_X, label=test_y, cat_features = categorical_indices)    
    print('Train CAT')
    model = CatBoostClassifier(**params)
    fit_model = model.fit(_train,
                          eval_set=_valid,
                          use_best_model=True,
                          verbose=1000,
                          plot=False)
    feature_im = fit_model.feature_importances_
    print('Predict 1/2')
    pred_test_y = fit_model.predict_proba(test_X)[:, 1]
    print('Predict 2/2')
    pred_test_y2 = fit_model.predict_proba(test_X2)[:, 1]
    return pred_test_y, pred_test_y2, feature_im


# Use some baseline parameters
cat_params = {'loss_function': 'CrossEntropy', 
              'eval_metric': "AUC",
              'task_type': "GPU",
              'learning_rate': 0.01,
              'iterations': 10000,
              'random_seed': 42,
              'od_type': "Iter",
#               'bagging_temperature': 0.2,
#               'depth': 10,
              'early_stopping_rounds': 500,
             }

n_folds = 5
results = run_cv_model(categorical_features_indices, train, test, target, runCAT, cat_params, auc, 'cat', n_folds=n_folds)

        
