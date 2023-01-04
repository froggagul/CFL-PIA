from data import load_yelp_author_with_attrs

x, y, prop = load_yelp_author_with_attrs('stars', 'user_id')
print(x.shape, y.shape, prop.shape)