from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Current block and month
BLOCK_NUM = 34
MONTH_DEF = 11

# load model
model = joblib.load('model/m1.pkl')
# load data
shop = pd.read_csv('data/shop_details.csv')

train = pd.read_csv('raw_data/prev_3_month.csv')
train = train[train['date_block_num'].isin(np.arange(BLOCK_NUM-3, BLOCK_NUM))]

train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month

nearest_prices_per_shop = pd.read_csv('data/nearest_prices_per_shop.csv')
mean_nearest_prices = pd.read_csv('data/mean_nearest_prices.csv')
mean_nearest_prices_per_cate = pd.read_csv('data/mean_nearest_prices_per_cate.csv')

# features
item_cate = pd.read_csv("data/nof_items_per_cate.csv")
avg_price = pd.read_csv('data/item_avg_price_per_shop.csv')
std_price = pd.read_csv('data/item_std_price_per_shop.csv')


features = ['month',
 'nof_items_per_cate',
 'item_price_rolling_mean_3',
 'item_price_rolling_std_3',
 'item_price_ratio_among_shops',
 'item_price_ratio_among_cate']
 
#  route to handle both form display and form submission
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            item_id = int(request.form['item_id'])
        except ValueError:
            error_message = "Please enter a valid integer for Item ID."
            return render_template('index.html', error_message=error_message)
        
        # Assuming you have a function or method to predict results
        results = predict_item_forecast(item_id)  # Implement your prediction logic

        # Render index.html with results
        return render_template('index.html', results=results, item_id=item_id)
    
    # Render initial form in index.html
    return render_template('index.html')

def predict_item_forecast(item_id):
    
    # 1. Filter data for the given item_id
    user = shop[shop['item_id'] == item_id].copy()
    
    # 2. Map neareast price to item for feature engineering
    user = user.merge(nearest_prices_per_shop, how='left')
    user = user.merge(mean_nearest_prices, how='left')
    user = user.merge(mean_nearest_prices_per_cate, how='left')

    # 3. Handle missing values in item_price
    user['item_price'] = user['item_price'].fillna(user['mean_nearest_price'])
    user['item_price'] = user['item_price'].fillna(user['mean_nearest_price_per_cate'])
    user['item_price'] = user['item_price'].fillna(300)  # based on the item price EDA median

    # 4. Set date_block_num and month
    user['date_block_num'] = BLOCK_NUM
    user['month'] = MONTH_DEF

    # 5. Merge with additional features for the test set
    user = user.merge(item_cate)

    # 6. Calculate item_price related features among shops
    item_price = user.groupby(['item_id', 'date_block_num'])['item_price'].mean().rename('item_avg_price_per_month').reset_index()

    user = user.merge(item_price)
    user['item_price_ratio_among_shops'] = user['item_price'] / user['item_avg_price_per_month']

    # 7. Calculate item_price related features among categories
    cate_price = user.groupby(['item_category_id', 'date_block_num'])['item_price'].mean().rename('cate_avg_price_per_month').reset_index()
    user = user.merge(cate_price)
    user['item_price_ratio_among_cate'] = user['item_price'] / user['cate_avg_price_per_month']

    # 8. Prepare rolling price features
    prev_data = train.groupby(['date_block_num', 'month', 'shop_id', 'item_id', 'item_category_id']).agg({'item_price': 'mean'}).reset_index()
    user_features = pd.concat([prev_data, user]).reset_index(drop=True)
    user_features.sort_values(by='date_block_num', inplace=True)

    user_features['item_price_rolling_mean_3'] = user_features.groupby(['shop_id', 'item_id'])['item_price'].rolling(window=3, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
    user_features['item_price_rolling_std_3'] = user_features.groupby(['shop_id', 'item_id'])['item_price'].rolling(window=3, min_periods=1).std().reset_index(level=[0, 1], drop=True)

    user_features = user_features.merge(avg_price, how='left')
    user_features['item_price_rolling_mean_3'].fillna(user_features['item_avg_price_per_shop'], inplace=True)
    user_features = user_features.merge(std_price, how='left')
    user_features['item_price_rolling_std_3'].fillna(user_features['item_std_price_per_shop'], inplace=True)

    user_features = user_features[user_features['date_block_num'] == BLOCK_NUM]

    # 9. Predict
    X_user = user_features[features]
    predictions = model.predict(X_user)

    # 10. Assign predictions to user dataframe
    user['prediction'] = predictions

    results = user[['shop_id', 'item_id', 'prediction']].sort_values(by=['shop_id'])
    results['prediction'] = results['prediction'].round(2)

    return results.to_dict(orient='records')


if __name__ == '__main__':
    app.run(debug=True)
